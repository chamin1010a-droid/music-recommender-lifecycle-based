import os
"""
[과제 4] Nostalgia 엔진

두 가지 Nostalgia 소스:
  1. 내부 Nostalgia: 내 기록에 있지만 오래 안 들은 곡 (161곡 풀)
  2. 외부 Nostalgia: "98년생이면 이것도 알지?" — Last.fm 인기곡 기반 추측

공통 규칙:
  - 현재 플레이리스트 장르/분위기와 매칭되는 곡만 삽입
  - 항상 넣지 않음 — 맥락이 맞을 때만 서프라이즈로
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
from lastfm_client import LastFMClient
from song_matcher import SongMatcher, normalize_artist


class NostalgiaEngine:
    """추억의 곡을 맥락에 맞게 서프라이즈하는 엔진"""
    
    # 98년생의 감수성 시기에 인기있었을 장르 태그
    NOSTALGIA_TAGS = [
        'korean ballad', 'k-pop', 'k-ballad',
        'korean pop', 'korean indie',
        'pop', 'pop rock', 'indie pop',
        'rock', 'alternative rock',
        'j-pop', 'j-rock',
    ]
    
    def __init__(self, api_key, song_temps, csv_path, birth_year=1998):
        self.lastfm = LastFMClient(api_key=api_key)
        self.song_temps = song_temps
        self.birth_year = birth_year
        self.matcher = SongMatcher(song_temps)
        
        # 감수성 시기: 10세~20세
        self.formative_start = birth_year + 10  # 2008
        self.formative_end = birth_year + 20    # 2018
        
        # 원본 데이터에서 재생 날짜 정보 추출
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        self.data_end = df['timestamp'].max()
        self.data_start = df['timestamp'].min()
        
        self.song_dates = {}
        for _, row in df.iterrows():
            sid = row.get('song_id', '')
            ts = row['timestamp']
            if sid not in self.song_dates:
                self.song_dates[sid] = {'first': ts, 'last': ts}
            else:
                if ts < self.song_dates[sid]['first']:
                    self.song_dates[sid]['first'] = ts
                if ts > self.song_dates[sid]['last']:
                    self.song_dates[sid]['last'] = ts
        
        # 현재 활발히 듣는 아티스트 (최근 3개월)
        recent = df[df['timestamp'] >= self.data_end - timedelta(days=90)]
        recent_counts = Counter()
        for _, row in recent.iterrows():
            a = str(row.get('artist', '')).replace(' - Topic', '').strip().lower()
            recent_counts[a] += 1
        self.active_artists = set(a for a, c in recent_counts.items() if c >= 10)
    
    def get_internal_nostalgia(self, n=20):
        """
        [소스 1] 내 기록에서 Nostalgia 후보 추출.
        초반 등장 + 오래 미재생 + 비활성 아티스트.
        """
        early_cutoff = self.data_start + timedelta(days=180)
        
        candidates = []
        for sid, info in self.song_temps.items():
            plays = info.get('total_plays', 0)
            artist = str(info.get('artist', '')).replace(' - Topic', '').strip()
            title = str(info.get('title', ''))
            
            if sid not in self.song_dates:
                continue
            
            first_play = self.song_dates[sid]['first']
            last_play = self.song_dates[sid]['last']
            days_since = (self.data_end - last_play).days
            
            # 현재 활발한 아티스트 곡은 Nostalgia가 아닌 Deep Dive 영역
            if artist.lower() in self.active_artists:
                continue
            
            # 초반 6개월 등장 + 3~15회 재생 + 6개월+ 미재생
            if first_play <= early_cutoff and 3 <= plays <= 15 and days_since >= 180:
                # 태그 수집 (장르 매칭용)
                tags = self.lastfm.get_combined_tags(artist, title)
                
                candidates.append({
                    'song_id': sid,
                    'artist': artist,
                    'title': title,
                    'plays': plays,
                    'days_since': days_since,
                    'tags': tags,
                    'source': 'internal',
                    'reason': f'{days_since}일간 미재생',
                })
        
        candidates.sort(key=lambda x: x['days_since'], reverse=True)
        return candidates[:n]
    
    def get_external_nostalgia(self, n=30):
        """
        [소스 2] "98년생이면 이것도 알지?" — Last.fm 인기곡 기반 추측.
        
        전략: 
        - 감수성 시기(2008~2018)에 인기했을 태그별 인기곡 수집
        - 내 라이브러리로 아는 아티스트(이미 듣는 가수)의 다른 히트곡은 알 확률 높음
        - 내 라이브러리에 없는 곡 중 "어 이것도 알아?" 급을 추천
        """
        print("  🕰️ '너 이것도 알지?' 후보 수집 중...")
        
        # 내가 아는 아티스트 목록 (전체 아티스트)
        my_artist_names = set()
        for sid, info in self.song_temps.items():
            a = str(info.get('artist', '')).replace(' - Topic', '').strip().lower()
            if a and a != 'nan':
                my_artist_names.add(a)
        
        candidates = []
        seen = set()
        
        # 방법 A: 내가 아는 아티스트의 Last.fm 인기곡 중 내가 안 들은 곡
        # → "이 가수 이 노래도 알지?" 
        top_artists_by_plays = Counter()
        for sid, info in self.song_temps.items():
            a = str(info.get('artist', '')).replace(' - Topic', '').strip()
            top_artists_by_plays[a] += info.get('total_plays', 0)
        
        # 발라드/팝 계열 아티스트를 우선 (Nostalgia 느낌)
        for artist, _ in top_artists_by_plays.most_common(30):
            tracks = self.lastfm.get_artist_top_tracks(artist, limit=10)
            for t in tracks:
                key = f"{t['artist'].lower()}||{t['name'].lower()}"
                if key in seen:
                    continue
                seen.add(key)
                
                # 내 라이브러리에 이미 있는 곡은 제외
                if self.matcher.is_in_library(t['artist'], t['name']):
                    continue
                
                tags = self.lastfm.get_combined_tags(t['artist'], t['name'])
                
                candidates.append({
                    'artist': t['artist'],
                    'title': t['name'],
                    'listeners': t.get('listeners', 0),
                    'tags': tags,
                    'source': 'artist_deep',
                    'reason': f'{artist}의 다른 히트곡',
                })
        
        # 방법 B: 장르 태그별 인기곡에서 내가 모를 수도 있는 명곡
        for tag in self.NOSTALGIA_TAGS:
            tracks = self.lastfm.get_tag_top_tracks(tag, limit=30)
            for t in tracks:
                key = f"{t['artist'].lower()}||{t['name'].lower()}"
                if key in seen:
                    continue
                seen.add(key)
                
                if self.matcher.is_in_library(t['artist'], t['name']):
                    continue
                
                # 내가 아는 아티스트의 곡이면 가산점
                knows_artist = t['artist'].lower() in my_artist_names
                
                tags = self.lastfm.get_combined_tags(t['artist'], t['name'])
                
                candidates.append({
                    'artist': t['artist'],
                    'title': t['name'],
                    'listeners': 0,
                    'tags': tags,
                    'source': 'era_hit',
                    'reason': f'{tag} 장르 명곡' + (' (아는 가수)' if knows_artist else ''),
                    'knows_artist': knows_artist,
                })
        
        # 아는 아티스트 곡 우선 정렬
        candidates.sort(key=lambda x: (
            x.get('knows_artist', False),
            x.get('listeners', 0)
        ), reverse=True)
        
        print(f"  → 외부 Nostalgia 후보 {len(candidates)}곡 수집 완료")
        return candidates[:n]
    
    def select_nostalgia_for_playlist(self, playlist_songs, n=1, seed_tracks=None):
        """
        현재 플레이리스트의 분위기에 맞는 Nostalgia 곡 선택.
        
        playlist_songs: 현재 플레이리스트의 곡 목록 (song_temps 형식)
        n: 선택할 Nostalgia 곡 수 (기본 1곡)
        seed_tracks: 시작곡 목록 (배정된 경우, 장르 매칭에 가중치 크게 반영)
        """
        # 현재 플레이리스트의 태그 프로필 수집
        playlist_tags = []
        for song in playlist_songs:
            tags = self.lastfm.get_combined_tags(
                song.get('artist', ''), song.get('title', '')
            )
            if tags:
                playlist_tags.extend(tags)
                
        # 시작곡 태그 강력 반영 (가중치 3배)
        if seed_tracks:
            for st in seed_tracks:
                st_tags = self.lastfm.get_combined_tags(st.get('artist', ''), st.get('title', ''))
                if st_tags:
                    # 가중치를 주기 위해 여러 번 리스트에 추가
                    for _ in range(3):
                        playlist_tags.extend(st_tags)
        
        
        if not playlist_tags:
            return []
        
        # 플레이리스트의 대표 태그 (상위 빈도)
        tag_freq = Counter(playlist_tags)
        top_tags = set(t for t, _ in tag_freq.most_common(10))
        
        # 내부 + 외부 Nostalgia 합산
        internal = self.get_internal_nostalgia(n=30)
        external = self.get_external_nostalgia(n=30)
        all_candidates = internal + external
        
        # 태그 매칭 점수 계산 (장르 매칭 기준 강화)
        scored = []
        for c in all_candidates:
            c_tags = set(c.get('tags', []))
            if not c_tags:
                continue
            
            # 현재 플레이리스트와의 태그 교집합
            overlap = len(c_tags & top_tags)
            if overlap < 2:
                continue  # 최소 태그 2개 이상 겹쳐야 장르 맞는 것으로 판단
            
            match_score = overlap / max(len(top_tags), 1)
            
            # 보너스: 아는 아티스트 +0.15, 내부 Nostalgia(확실히 아는 곡) +0.1
            bonus = 0
            if c.get('knows_artist', False):
                bonus += 0.15
            if c.get('source') == 'internal':
                bonus += 0.1
            
            c['nostalgia_score'] = match_score + bonus
            scored.append(c)
        
        scored.sort(key=lambda x: x['nostalgia_score'], reverse=True)
        return scored[:n]
    
    def display_nostalgia_pool(self, internal, external):
        """Nostalgia 풀 전체를 보기 좋게 출력"""
        print(f"\n{'=' * 70}")
        print("🕰️ Nostalgia 후보 풀")
        print(f"{'=' * 70}")
        
        print(f"\n  📦 내부 (내 기록에서): {len(internal)}곡")
        for i, s in enumerate(internal[:10], 1):
            tags_str = ', '.join(s.get('tags', [])[:3]) if s.get('tags') else '태그 없음'
            print(f"    {i:>2}. [{s['plays']:>2}회, {s['days_since']}일전] "
                  f"{s['artist']:<20} — {s['title'][:30]}")
            print(f"        태그: {tags_str}")
        
        if len(internal) > 10:
            print(f"    ... 외 {len(internal)-10}곡")
        
        print(f"\n  🌐 외부 ('너 이것도 알지?'): {len(external)}곡")
        for i, s in enumerate(external[:10], 1):
            tags_str = ', '.join(s.get('tags', [])[:3]) if s.get('tags') else '태그 없음'
            src_label = '🎯' if s.get('knows_artist') else '🌍'
            print(f"    {i:>2}. {src_label} {s['artist']:<20} — {s['title'][:30]}")
            print(f"        {s['reason']} | 태그: {tags_str}")
        
        if len(external) > 10:
            print(f"    ... 외 {len(external)-10}곡")
        
        print(f"\n  총 Nostalgia 풀: {len(internal) + len(external)}곡")
        print(f"{'=' * 70}")


# =============================================================================
# 독립 실행 테스트
# =============================================================================
if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    from lifecycle_recommender import run_pipeline
    
    csv_path = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
    meta_path = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\ytm_metadata_cache.csv'
    
    API_KEY = os.environ.get("LASTFM_API_KEY", "")
    
    result = run_pipeline(csv_path, 'user', 15, 'default', meta_path, 1998)
    song_temps = result['temp_tracker'].song_temps
    playlist = result['playlist']
    
    # Nostalgia 엔진 실행
    engine = NostalgiaEngine(
        api_key=API_KEY,
        song_temps=song_temps,
        csv_path=csv_path,
        birth_year=1998
    )
    
    # 전체 풀 확인
    internal = engine.get_internal_nostalgia(n=30)
    external = engine.get_external_nostalgia(n=30)
    engine.display_nostalgia_pool(internal, external)
    
    # 현재 플레이리스트에 맞는 Nostalgia 선택
    print(f"\n🎯 현재 플레이리스트에 어울리는 Nostalgia 추천:")
    selected = engine.select_nostalgia_for_playlist(playlist, n=3)
    for i, s in enumerate(selected, 1):
        score = s.get('nostalgia_score', 0)
        src = '내 기록' if s.get('source') == 'internal' else '추측'
        print(f"  {i}. {s['artist']} — {s['title'][:35]}")
        print(f"     [{src}] {s.get('reason', '')} | 매칭점수: {score:.2f}")
