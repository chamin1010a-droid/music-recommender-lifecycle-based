import os
"""
[과제 3] 신곡 발굴 엔진 v2 (External Discovery Pipeline)

3단계 Discovery 분류:
  🔵 Deep Dive  : 꽤 아는 가수(11~29곡)의 안 들어본 곡
  🟢 Expand     : 조금 아는 가수(3~10곡)의 안 들어본 곡 
  🟡 Explore    : 거의 모르는 가수(0~2곡)의 곡

기본 비중: Deep Dive 40% / Expand 40% / Explore 20%

곡 매칭에 SongMatcher를 사용하여 영문/한글 제목 불일치 버그를 근본적으로 해결.
"""

import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from lastfm_client import LastFMClient
from song_matcher import SongMatcher


class ExternalDiscoveryEngine:
    """내 라이브러리 밖에서 신곡을 발굴하는 엔진"""
    
    # 3단계 Discovery 비중 프리셋
    DISCOVERY_PRESETS = {
        'default':   {'deep_dive': 0.40, 'expand': 0.40, 'explore': 0.20},
        'safe':      {'deep_dive': 0.60, 'expand': 0.30, 'explore': 0.10},
        'adventure': {'deep_dive': 0.20, 'expand': 0.30, 'explore': 0.50},
    }
    
    def __init__(self, api_key, song_temps, similarity_engine=None):
        self.lastfm = LastFMClient(api_key=api_key)
        self.song_temps = song_temps
        self.similarity_engine = similarity_engine
        
        # SongMatcher로 통합 매칭 (괄호 추출 + 퍼지 매칭 + alias 매핑)
        self.matcher = SongMatcher(song_temps)
    
    def _get_top_artists(self, n=15):
        """내 라이브러리에서 재생 횟수 기준 TOP 아티스트 추출"""
        artist_plays = Counter()
        for sid, info in self.song_temps.items():
            artist = info.get('artist', '')
            plays = info.get('total_plays', 0)
            artist_plays[artist] += plays
        return [artist for artist, _ in artist_plays.most_common(n)]
    
    def _classify_familiarity(self, artist):
        """
        아티스트 친숙도를 3단계로 분류.
        Returns: ('deep_dive' | 'expand' | 'explore', 보유곡수)
        """
        count = self.matcher.get_artist_known_count(artist, min_plays=5)
        if count >= 11:
            return 'deep_dive', count
        elif count >= 3:
            return 'expand', count
        else:
            return 'explore', count

    def _calculate_tag_similarity_to_library(self, artist, track):
        """외부 곡의 태그와 내 라이브러리 평균 태그 벡터 간 유사도 (0~1)"""
        candidate_tags = self.lastfm.get_combined_tags(artist, track)
        if not candidate_tags:
            return 0.0
        
        my_top_songs = sorted(
            self.song_temps.values(), 
            key=lambda x: x.get('total_plays', 0), 
            reverse=True
        )[:50]
        
        my_tag_docs = []
        for s in my_top_songs:
            tags = self.lastfm.get_combined_tags(s.get('artist', ''), s.get('title', ''))
            if tags:
                my_tag_docs.append(" ".join(tags))
        
        if not my_tag_docs:
            return 0.0
        
        all_docs = my_tag_docs + [" ".join(candidate_tags)]
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(all_docs)
        
        candidate_vec = tfidf[-1]
        library_vecs = tfidf[:-1]
        
        similarities = cosine_similarity(candidate_vec, library_vecs)[0]
        return float(np.mean(similarities))

    def discover_new_songs(self, n=15, min_similarity=0.25, max_similarity=0.85,
                           discovery_preset='default',
                           seed_tracks=None, seed_vector=None, similarity_engine=None):
        """
        신곡 발굴 메인 함수.
        
        3단계 Discovery를 비중에 맞게 균형 있게 추천합니다.
        """
        preset = self.DISCOVERY_PRESETS.get(discovery_preset, self.DISCOVERY_PRESETS['default'])
        
        print(f"\n🔍 [신곡 발굴 v2] 시작... (프리셋: {discovery_preset})")
        print(f"  비중: Deep Dive {preset['deep_dive']*100:.0f}% / "
              f"Expand {preset['expand']*100:.0f}% / "
              f"Explore {preset['explore']*100:.0f}%")
        
        # 1단계: 유사 아티스트 탐색
        if seed_tracks:
            top_artists = list(set([t.get('artist', '').replace(' - Topic', '').strip() for t in seed_tracks if t.get('artist')]))
            print(f"  시작곡 아티스트({len(top_artists)}명)에서 유사 가수 탐색 중...")
        else:
            top_artists = self._get_top_artists(n=15)
            print(f"  내 TOP 아티스트 {len(top_artists)}명에서 유사 가수 탐색 중...")
        
        similar_artists = {}
        for my_artist in top_artists:
            clean = my_artist.replace(' - Topic', '').strip()
            similars = self.lastfm.get_similar_artists(clean, limit=15)
            for s in similars:
                name = s['name'].lower()
                fam_type, fam_count = self._classify_familiarity(name)
                # 30곡 이상 아는 가수는 제외
                if fam_count >= 30:
                    continue
                if name not in similar_artists or s['match'] > similar_artists[name]['match']:
                    similar_artists[name] = {
                        'name': s['name'],
                        'match': s['match'],
                        'source': clean,
                        'fam_type': fam_type,
                        'fam_count': fam_count,
                    }
        
        print(f"  → 유사 가수 {len(similar_artists)}명 발견")
        
        # 2단계: 대표곡 수집 + SongMatcher 중복 체크
        candidates = {'deep_dive': [], 'expand': [], 'explore': []}
        artist_count = 0
        sorted_artists = sorted(similar_artists.values(), key=lambda x: x['match'], reverse=True)
        
        for artist_info in sorted_artists[:50]:
            tracks = self.lastfm.get_artist_top_tracks(artist_info['name'], limit=5)
            for t in tracks:
                # SongMatcher로 정밀 중복 체크 (괄호/퍼지매칭 포함!)
                if not self.matcher.is_in_library(t['artist'], t['name']):
                    entry = {
                        'artist': t['artist'],
                        'track': t['name'],
                        'listeners': t.get('listeners', 0),
                        'source_artist': artist_info['source'],
                        'artist_match': artist_info['match'],
                        'fam_type': artist_info['fam_type'],
                        'fam_count': artist_info['fam_count'],
                    }
                    candidates[artist_info['fam_type']].append(entry)
            artist_count += 1
            if artist_count % 10 == 0:
                print(f"  대표곡 수집 중... {artist_count}/{min(50, len(sorted_artists))} 아티스트")
        
        for k, v in candidates.items():
            print(f"  → {k}: {len(v)}곡")
        
        # 3단계: 태그 유사도 필터링
        print(f"  태그 유사도 필터링 중...")
        for tier in candidates:
            filtered = []
            for c in candidates[tier]:
                sim = self._calculate_tag_similarity_to_library(c['artist'], c['track'])
                c['tag_similarity'] = sim
                if min_similarity <= sim <= max_similarity:
                    filtered.append(c)
            candidates[tier] = filtered
        
        for k, v in candidates.items():
            print(f"  → {k} (필터 후): {len(v)}곡")
        
        # 4단계: 각 Tier 내에서 점수 계산 & 정렬
        for tier in candidates:
            songs = candidates[tier]
            if not songs:
                continue
            max_listeners = max(c.get('listeners', 1) for c in songs)
            for c in songs:
                pop_score = min(c.get('listeners', 0) / max(max_listeners, 1), 1.0)
                c['final_score'] = (
                    c['tag_similarity'] * 0.5 +
                    c['artist_match'] * 0.3 +
                    pop_score * 0.2
                )
            songs.sort(key=lambda x: x['final_score'], reverse=True)
            
            # 아티스트당 2곡 제한
            deduped = []
            ac = Counter()
            for c in songs:
                a = c['artist'].lower()
                if ac[a] < 2:
                    deduped.append(c)
                    ac[a] += 1
            candidates[tier] = deduped
        
        # 5단계: 비중에 맞게 3-tier 합산
        n_deep = max(1, round(n * preset['deep_dive']))
        n_expand = max(1, round(n * preset['expand']))
        n_explore = max(1, n - n_deep - n_expand)
        
        result = []
        result.extend(candidates['deep_dive'][:n_deep])
        result.extend(candidates['expand'][:n_expand])
        result.extend(candidates['explore'][:n_explore])
        
        # 부족하면 다른 tier에서 보충
        if len(result) < n:
            remaining = n - len(result)
            all_remaining = []
            for tier in candidates:
                used = len([r for r in result if r.get('fam_type') == tier])
                all_remaining.extend(candidates[tier][used:])
            all_remaining.sort(key=lambda x: x.get('final_score', 0), reverse=True)
            result.extend(all_remaining[:remaining])
        
        print(f"\n✅ [신곡 발굴 v2] 완료! 최종 추천 {len(result)}곡")
        return result[:n]
    
    def display_recommendations(self, recommendations):
        """발굴된 신곡 목록을 3-tier 분류 포함하여 출력"""
        if not recommendations:
            print("  추천할 신곡이 없습니다.")
            return
        
        tier_icons = {
            'deep_dive': '🔵', 'expand': '🟢', 'explore': '🟡'
        }
        tier_labels = {
            'deep_dive': 'Deep Dive (꽤 아는 가수)',
            'expand': 'Expand (조금 아는 가수)',
            'explore': 'Explore (새로운 가수)'
        }
        
        print(f"\n{'=' * 70}")
        print("🆕 신곡 추천 — 3단계 Discovery")
        print(f"{'=' * 70}")
        
        for i, r in enumerate(recommendations, 1):
            sim_pct = r.get('tag_similarity', 0) * 100
            fam_type = r.get('fam_type', 'explore')
            fam_count = r.get('fam_count', 0)
            source = r.get('source_artist', '?')
            icon = tier_icons.get(fam_type, '⚪')
            
            fam_label = f"보유 {fam_count}곡" if fam_count > 0 else "신규"
            
            bar_len = int(sim_pct / 5)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            
            print(f"  {i:>2}. {icon} {r['artist']:<20} — {r['track'][:30]}")
            print(f"       태그: |{bar}| {sim_pct:.0f}%  "
                  f"{fam_label}  (← {source})")
        
        # Tier별 통계
        print(f"\n--- Discovery 구성 ---")
        tier_counts = Counter(r.get('fam_type', 'explore') for r in recommendations)
        for tier, label in tier_labels.items():
            cnt = tier_counts.get(tier, 0)
            print(f"  {tier_icons[tier]} {label}: {cnt}곡")
        
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
    
    result = run_pipeline(
        csv_path=csv_path,
        user_name='user',
        playlist_size=15,
        preset='default',
        metadata_path=meta_path,
        user_birth_year=1998
    )
    
    song_temps = result['temp_tracker'].song_temps
    
    engine = ExternalDiscoveryEngine(
        api_key=API_KEY,
        song_temps=song_temps
    )
    
    recommendations = engine.discover_new_songs(n=15, discovery_preset='default')
    engine.display_recommendations(recommendations)
