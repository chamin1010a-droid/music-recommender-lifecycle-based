"""
검색 기록 파싱 + 능동적 재생 판별 스크립트
=============================================
Google Takeout의 '검색 기록.html'을 파싱하여,
재생 기록(features CSV)과 시간 매칭함으로써
각 재생이 "능동적(검색 후 재생)"인지 판별합니다.

핵심 로직:
  1. 검색 기록 HTML 파싱 → (검색어, 시각, 플랫폼) 추출
  2. YouTube Music 검색만 필터링
  3. 재생 기록의 각 row에서, 직전 10분 이내에 해당 곡/아티스트를 검색한 기록이 있으면 '능동적 재생'으로 판정
  4. 매칭 전략 (3중 안전장치):
     - Level 1: 정규화된 아티스트명 매칭 (한글↔영문 별칭 포함)
     - Level 2: 곡 제목 매칭 (괄호 안 부제, 정규화)
     - Level 3: 퍼지 매칭 (부분 문자열 + 유사도 70% 이상)
"""

import os
import re
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from urllib.parse import unquote
from difflib import SequenceMatcher

# ===================================================================
# 1. 아티스트 한글↔영문 별칭 사전 (매칭 버그 방지의 핵심)
# ===================================================================
# 검색은 '잔나비'로 하지만, 재생 기록에는 'JANNABI - Topic'으로 기록됨
# 이 사전이 없으면 매칭이 깨짐
ARTIST_ALIASES = {
    # 한글 → 영문 (검색어에서 쓰는 이름 → 재생기록의 이름)
    '잔나비': ['jannabi', 'jannabi - topic'],
    '검정치마': ['the black skirts', 'the black skirts - topic'],
    '한로로': ['hanroro', 'hanroro - topic'],
    '카더가든': ['car, the garden', 'car the garden', 'car, the garden - topic'],
    '장범준': ['jang beom june', 'jang beom june - topic'],
    '로이킴': ['roy kim', 'roy kim - topic'],
    '쏜애플': ['thornapple', 'thornapple - topic'],
    '데이식스': ['day6', 'day6 - topic'],
    '뮤즈': ['muse', 'muse - topic'],
    '오아시스': ['oasis', 'oasis - topic'],
    '실리카겔': ['silica gel', 'silicagel'],
    '웅키': ['ungki', 'ungki - topic'],
    '악뮤': ['akmu', 'akmu - topic'],
    '악동뮤지션': ['akmu', 'akmu - topic'],
    '볼빨간사춘기': ['bol4', 'bolbbalgan4'],
    '아이유': ['iu', 'iu - topic'],
    '방탄소년단': ['bts', 'bts - topic'],
    '엑스디너리히어로즈': ['xdinary heroes', 'xdinary heroes - topic'],
    '더볼런티어즈': ['the volunteers', 'the volunteers - topic'],
    '데이먼스이어': ['damons year', 'damons year - topic'],
    '찰리푸스': ['charlie puth', 'charlie puth - topic'],
    '마룬파이브': ['maroon 5', 'maroon 5 - topic'],
    '버즈': ['buzz', 'buzz - topic'],
    '디셈버': ['december', 'december - topic'],
    '먼데이키즈': ['monday kiz', 'monday kiz - topic'],
    '박효신': ['park hyo shin', 'park hyo shin - topic'],
    '성시경': ['sung si-kyung', 'sung si kyung', 'sung si-kyung - topic'],
    '김종국': ['kim jong kook', 'kim jong kook - topic'],
    '최유리': ['choi yu ree', 'choi yu ree - topic'],
    'sg워너비': ['sg wannabe', 'sg wannabe - topic'],
    '엠씨더맥스': ['m.c the max', 'mc the max', 'm.c the max - topic'],
    '원리퍼블릭': ['onerepublic', 'onerepublic - topic'],
    '마이케미컬로맨스': ['my chemical romance', 'my chemical romance - topic'],
    '봉주르레코드': ['bonjour records'],
    '로제': ['rosé', 'rose', 'rosé - topic'],
    '라디오헤드': ['radiohead', 'radiohead - topic'],
    '너드커넥션': ['nerd connection', 'nerd connection - topic'],
    '더일렉트릭이얼즈': ['the electriceels', 'the electriceels - topic'],
    '밴드나': ['band nah', 'band nah - topic'],
}

# 영문 약식 → 정규화 (검색에서 약식으로 칠 수 있음)
ENGLISH_ALIASES = {
    'mcr': ['my chemical romance', 'my chemical romance - topic'],
    'day6': ['day6 - topic'],
    'xdh': ['xdinary heroes', 'xdinary heroes - topic'],
    'xdinary heroes': ['xdinary heroes - topic'],
    'jannabi': ['jannabi - topic'],
    'black skirts': ['the black skirts', 'the black skirts - topic'],
    'hanroro': ['hanroro - topic'],
    'oasis': ['oasis - topic'],
    'muse': ['muse - topic'],
    'charlie puth': ['charlie puth - topic'],
    'maroon 5': ['maroon 5 - topic'],
}


def normalize_text(text):
    """매칭을 위한 텍스트 정규화: 소문자, 특수문자 제거, 공백 정리"""
    if not text or not isinstance(text, str):
        return ''
    text = text.lower().strip()
    # HTML 엔티티 해결
    text = text.replace('&amp;', '&').replace('&#39;', "'").replace('&quot;', '"')
    # " - Topic" 접미사 제거 (매칭 시에만)
    text = re.sub(r'\s*-\s*topic$', '', text)
    # 괄호 안 내용 추출용 (나중에 따로 비교)
    # 특수문자 정리
    text = re.sub(r'[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ\-\']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_bracket_content(text):
    """괄호 안의 부제를 추출: '좋은 밤 좋은 꿈 (Good Night Good Dream)' → 'good night good dream'"""
    if not text:
        return []
    matches = re.findall(r'[(\[](.*?)[)\]]', text)
    return [normalize_text(m) for m in matches if len(m) > 2]


def get_artist_search_names(artist_field):
    """재생 기록의 아티스트 필드에서 검색 가능한 이름 변형 목록을 생성"""
    names = set()
    if not artist_field or not isinstance(artist_field, str):
        return names
    
    norm = normalize_text(artist_field)
    names.add(norm)
    
    # " - Topic" 없는 버전
    clean = re.sub(r'\s*-\s*topic$', '', norm).strip()
    if clean:
        names.add(clean)
    
    return names


# ===================================================================
# 2. 검색 기록 HTML 파싱
# ===================================================================
def parse_search_history(html_path):
    """
    Google Takeout 검색 기록 HTML을 파싱하여 
    (search_query, timestamp, platform) 리스트를 반환한다.
    """
    print(f"[1/4] 검색 기록 HTML 파싱 중: {html_path}")
    
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()
    
    # 레코드 단위로 분리
    chunks = html.split('class="outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp">')
    
    records = []
    for chunk in chunks[1:]:  # 첫 번째는 헤더
        # 플랫폼 판별
        platform_match = re.search(r'<p class="mdl-typography--title">\s*(YouTube Music|YouTube)\s*<br>', chunk)
        if not platform_match:
            continue
        platform = platform_match.group(1).strip()
        
        # 검색어 추출 (URL에서 + search_query 디코딩)
        query_match = re.search(
            r'<a[^>]*href="https://www\.youtube\.com/results\?search_query=([^"]+)"[^>]*>([^<]+)</a>',
            chunk
        )
        if not query_match:
            continue
        
        # URL 인코딩된 검색어와 표시용 검색어 둘 다 추출
        url_query = unquote(query_match.group(1).replace('+', ' ')).strip()
        display_query = query_match.group(2).strip()
        
        # 시각 추출
        # 패턴: 2024. 10. 19. AM 9:15:07 KST 또는 2024. 10. 19. 오전 9:15:07 KST
        time_match = re.search(
            r'(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.\s*(오전|오후|AM|PM)\s*(\d{1,2}):(\d{2}):(\d{2})',
            chunk
        )
        if not time_match:
            continue
            
        y, m, d, ampm, h, mi, sec = time_match.groups()
        h = int(h)
        ampm_norm = ampm.replace('AM', '오전').replace('PM', '오후')
        if ampm_norm == '오후' and h < 12:
            h += 12
        elif ampm_norm == '오전' and h == 12:
            h = 0
        
        try:
            ts = datetime(int(y), int(m), int(d), h, int(mi), int(sec))
        except ValueError:
            continue
        
        records.append({
            'search_query': display_query,      # 기본 표시용 검색어
            'search_query_url': url_query,       # URL 디코딩된 검색어 (더 정확)
            'timestamp': ts,
            'platform': platform,
        })
    
    print(f"  → 전체 검색 기록: {len(records)}건")
    
    # YouTube Music만 필터링
    music_records = [r for r in records if r['platform'] == 'YouTube Music']
    print(f"  → YouTube Music 검색: {len(music_records)}건 (일반 YouTube {len(records) - len(music_records)}건 제외)")
    
    return music_records


# ===================================================================
# 3. 검색어 ↔ 재생 기록 매칭 엔진
# ===================================================================
class SearchPlayMatcher:
    """
    검색 기록과 재생 기록을 시간+텍스트 매칭하여 '능동적 재생'을 판별한다.
    
    매칭 전략 (3중 안전장치):
      Level 1: 아티스트명 직접 매칭 (별칭 사전 활용)
      Level 2: 곡 제목 직접 매칭 (정규화 + 부제 추출)
      Level 3: 퍼지 매칭 (부분 문자열 포함 or SequenceMatcher 70%+)
    """
    
    def __init__(self, time_window_minutes=10):
        self.time_window = timedelta(minutes=time_window_minutes)
        
        # 역방향 별칭 사전 구축: '잔나비' → ['jannabi', 'jannabi - topic']
        # 키를 정규화해서 저장
        self.alias_lookup = {}
        for kr_name, en_names in ARTIST_ALIASES.items():
            norm_kr = normalize_text(kr_name)
            self.alias_lookup[norm_kr] = [normalize_text(n) for n in en_names]
        for en_name, mapped in ENGLISH_ALIASES.items():
            norm_en = normalize_text(en_name)
            self.alias_lookup[norm_en] = [normalize_text(n) for n in mapped]
    
    def _resolve_search_to_targets(self, search_query):
        """
        검색어를 매칭 가능한 타겟 문자열 집합으로 확장한다.
        '잔나비' → {'잔나비', 'jannabi', 'jannabi - topic'}
        """
        norm_query = normalize_text(search_query)
        targets = {norm_query}
        
        # 별칭 사전에서 확장
        if norm_query in self.alias_lookup:
            targets.update(self.alias_lookup[norm_query])
        
        # 부분 매칭용: 2글자 이상이면 원본도 추가
        if len(norm_query) >= 2:
            targets.add(norm_query)
        
        return targets
    
    def _match_score(self, search_targets, artist_names, title_norm, title_brackets):
        """
        검색 타겟과 재생 기록(아티스트+제목)의 매칭 점수를 반환.
        0 = 매칭 안됨, 1 = Level3(퍼지), 2 = Level2(제목), 3 = Level1(아티스트)
        """
        
        # Level 1: 아티스트명 직접 매칭
        for target in search_targets:
            for artist_name in artist_names:
                if target == artist_name:
                    return 3  # 완전 일치
                # 아티스트명이 검색어를 포함하거나 그 반대 (2글자 이상일 때만)
                if len(target) >= 2 and (target in artist_name or artist_name in target):
                    return 3
        
        # Level 2: 곡 제목 직접 매칭
        for target in search_targets:
            if len(target) < 2:
                continue
            # 제목 완전 일치
            if target == title_norm:
                return 2
            # 제목 포함 (검색어가 3글자 이상일 때)
            if len(target) >= 3 and (target in title_norm or title_norm in target):
                return 2
            # 괄호 안 부제와 매칭
            for bracket in title_brackets:
                if target == bracket or (len(target) >= 3 and target in bracket):
                    return 2
        
        # Level 3: 퍼지 매칭 (SequenceMatcher) — 검색어가 4글자 이상일 때만
        for target in search_targets:
            if len(target) < 4:
                continue
            # 아티스트명과 퍼지 비교
            for artist_name in artist_names:
                ratio = SequenceMatcher(None, target, artist_name).ratio()
                if ratio >= 0.70:
                    return 1
            # 제목과 퍼지 비교
            ratio = SequenceMatcher(None, target, title_norm).ratio()
            if ratio >= 0.70:
                return 1
        
        return 0
    
    def match_searches_to_plays(self, search_records, play_df):
        """
        각 재생 기록에 대해, 직전 time_window 내에 관련 검색이 있었는지 판별.
        
        Returns:
            play_df에 'is_proactive', 'proactive_query', 'match_level' 컬럼 추가
        """
        print(f"\n[3/4] 검색-재생 매칭 중 (시간 윈도우: {self.time_window.seconds // 60}분)...")
        
        df = play_df.copy()
        df['play_timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 검색 기록을 시간순 정렬
        searches = sorted(search_records, key=lambda x: x['timestamp'])
        search_times = [s['timestamp'] for s in searches]
        
        # 결과 컬럼 초기화
        df['is_proactive'] = 0
        df['proactive_query'] = ''
        df['match_level'] = 0
        
        # 각 재생마다 전처리된 아티스트/제목 정보 캐싱
        play_info_cache = {}
        for idx, row in df.iterrows():
            artist = str(row.get('artist', ''))
            title = str(row.get('title', ''))
            
            artist_names = get_artist_search_names(artist)
            title_norm = normalize_text(title)
            title_brackets = extract_bracket_content(title)
            
            play_info_cache[idx] = (artist_names, title_norm, title_brackets)
        
        # 이진 탐색으로 시간 윈도우 안의 검색만 빠르게 찾기
        import bisect
        
        match_count = 0
        total = len(df)
        
        for i, (idx, row) in enumerate(df.iterrows()):
            if (i + 1) % 2000 == 0:
                print(f"  진행: {i+1}/{total} ({match_count}건 매칭됨)")
            
            play_time = row['play_timestamp']
            window_start = play_time - self.time_window
            
            # 이진 탐색: window_start ~ play_time 사이의 검색 인덱스 찾기
            left = bisect.bisect_left(search_times, window_start)
            right = bisect.bisect_right(search_times, play_time)
            
            if left >= right:
                continue  # 이 시간대에 검색 기록 없음
            
            # 시간 윈도우 안의 검색들과 매칭 시도
            artist_names, title_norm, title_brackets = play_info_cache[idx]
            
            best_score = 0
            best_query = ''
            
            for si in range(left, right):
                search = searches[si]
                # URL 디코딩된 쿼리와 표시용 쿼리 둘 다 시도
                for query_field in ['search_query_url', 'search_query']:
                    query = search[query_field]
                    search_targets = self._resolve_search_to_targets(query)
                    
                    score = self._match_score(search_targets, artist_names, title_norm, title_brackets)
                    if score > best_score:
                        best_score = score
                        best_query = search['search_query']
                
                if best_score >= 3:  # 이미 최고 점수면 더 볼 필요 없음
                    break
            
            if best_score > 0:
                df.at[idx, 'is_proactive'] = 1
                df.at[idx, 'proactive_query'] = best_query
                df.at[idx, 'match_level'] = best_score
                match_count += 1
        
        print(f"  → 총 {match_count}건 능동적 재생 판별 ({match_count/total*100:.1f}%)")
        
        return df


# ===================================================================
# 4. 간접 능동성 신호 감지 + 스트릭 기반 점수 산출
# ===================================================================
def detect_indirect_proactive_signals(df):
    """
    검색 기록 외에, 재생 패턴에서 '의도적 재생'을 추론한다.
    
    신호 3가지:
      1. 세션 시작곡: 30분+ 공백 후 첫 곡 (앱 열고 첫 선택)
      2. 아티스트 연속 청취 (Deep Dive): 같은 가수 4곡+ 연속 재생
         → "잔나비 앨범을 골라서 틀었다"
      3. 맥락 전환: 같은 가수 4곡+ 후 다른 가수로 전환
         → "잔나비 듣다가 내가 직접 검정치마를 골랐다"
    
    스트릭 길이 반영:
      - 같은 가수 4곡 vs 10곡은 다른 점수를 받음 (log₂ 체감)
      - 3곡까지는 우연 → 0점, 4곡부터 시작, 10곡+ 캡
    
    중복 방지:
      - 세션 시작 + Deep Dive가 동시 해당되면, 더 높은 쪽만 적용
    """
    print(f"\n[추가] 간접 능동성 신호 감지 중...")
    
    df = df.copy()
    df['play_ts'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('play_ts').reset_index(drop=True)
    
    SESSION_GAP_MINUTES = 30
    STREAK_MIN = 4  # 4곡 이상부터 의도적 선택으로 판정
    
    # === 시간 간격 계산 ===
    df['prev_play_ts'] = df['play_ts'].shift(1)
    df['gap_minutes'] = (df['play_ts'] - df['prev_play_ts']).dt.total_seconds() / 60
    
    # === 신호 1: 세션 시작곡 ===
    df['is_session_start'] = (df['gap_minutes'] >= SESSION_GAP_MINUTES).astype(int)
    df.loc[df.index[0], 'is_session_start'] = 1
    
    session_starts = df['is_session_start'].sum()
    print(f"  ▶️ 세션 시작곡: {session_starts}건 (30분+ 공백 후 첫 곡)")
    
    # === 아티스트 연속 스트릭 계산 ===
    df['artist_changed'] = (df['artist'] != df['artist'].shift(1)).astype(int)
    # 세션이 바뀌면(30분+ 공백) 스트릭도 끊김
    df.loc[df['is_session_start'] == 1, 'artist_changed'] = 1
    df['streak_id'] = df['artist_changed'].cumsum()
    streak_sizes = df.groupby('streak_id')['artist'].transform('count')
    df['streak_length'] = streak_sizes
    
    # === 신호 2: 아티스트 Deep Dive (같은 가수 4곡+ 연속) ===
    df['is_artist_deep_dive'] = (df['streak_length'] >= STREAK_MIN).astype(int)
    
    deep_dives = df[df['is_artist_deep_dive'] == 1]['streak_id'].nunique()
    deep_dive_plays = df['is_artist_deep_dive'].sum()
    print(f"  🎵 아티스트 연속 청취: {deep_dives}회 발생, {deep_dive_plays}곡 해당 (같은 가수 {STREAK_MIN}곡+ 연속)")
    
    # === 신호 3: 맥락 전환 (4곡+ 스트릭 후 전환) ===
    # 직전 스트릭 길이 가져오기
    streak_info = df.groupby('streak_id').agg(
        size=('artist', 'count'),
        last_idx=('artist', lambda x: x.index[-1])
    )
    
    df['is_context_switch'] = 0
    for sid in streak_info.index:
        if sid + 1 not in streak_info.index:
            continue
        prev_size = streak_info.loc[sid, 'size']
        next_first_idx = streak_info.loc[sid, 'last_idx'] + 1
        if next_first_idx in df.index and prev_size >= STREAK_MIN:
            # 세션이 안 바뀌었을 때만
            if df.loc[next_first_idx, 'is_session_start'] == 0:
                df.loc[next_first_idx, 'is_context_switch'] = 1
    
    context_switches = df['is_context_switch'].sum()
    print(f"  🔀 맥락 전환: {context_switches}건 (동일 가수 {STREAK_MIN}곡+ 후 전환)")
    
    # 임시 컬럼 정리
    df = df.drop(columns=['play_ts', 'prev_play_ts', 'gap_minutes',
                          'artist_changed', 'streak_id'])
    
    return df


def compute_proactive_score(df):
    """
    능동성 점수 산출 (스트릭 길이 체감 + 중복 방지)
    
    점수 구조:
      proactive_score = search_bonus + intent_bonus (최대 1.0)
      
      search_bonus: 검색 후 재생이면 +0.4 (직접적 증거)
      
      intent_bonus: 행동 신호 중 가장 강한 것 하나만 적용 (중복 방지)
        - 세션 시작곡:      0.3 × streak_scale
        - Deep Dive(4곡+):  0.3 × streak_scale  
        - 맥락 전환:        0.2 × streak_scale
        
      streak_scale = min(1.0, log₂(streak_length) / log₂(10))
        - 1곡: 0.00  → 세션 시작했지만 바로 나감
        - 4곡: 0.60  → 어느 정도 의도적
        - 8곡: 0.90  → 거의 확실히 의도적
        - 10곡+: 1.00 → 캡 (더 이상 안 올라감)
    
    예시:
      잔나비 재생목록에서 10곡 연속 청취 (검색 안 함):
        search=0 + deep_dive(0.3 × 1.0) = 0.30
      
      DAY6 검색 후 4곡 청취:
        search=0.4 + deep_dive(0.3 × 0.60) = 0.58
      
      세션 시작 후 1곡만 듣고 끔:
        search=0 + session(0.3 × 0.0) = 0.00  (의미 없는 시작)
      
      세션 시작 후 8곡 연속 같은 가수:
        search=0 + max(session, deep_dive) = 0.3 × 0.90 = 0.27
    """
    import math
    
    WEIGHT_SEARCH = 0.4
    WEIGHT_SESSION = 0.3
    WEIGHT_DEEP_DIVE = 0.3
    WEIGHT_SWITCH = 0.2
    
    def streak_scale(length):
        """스트릭 길이를 0~1 점수로 변환 (log₂ 체감)"""
        if length <= 1:
            return 0.0
        return min(1.0, math.log2(length) / math.log2(10))
    
    df = df.copy()
    
    # 각 play의 행동 신호별 점수 계산
    scores = []
    for idx, row in df.iterrows():
        search_bonus = WEIGHT_SEARCH if row['is_proactive'] == 1 else 0.0
        
        streak_len = row.get('streak_length', 1)
        scale = streak_scale(streak_len)
        
        # 행동 신호 점수 (각각 계산)
        intent_candidates = []
        
        if row['is_session_start'] == 1:
            intent_candidates.append(WEIGHT_SESSION * scale)
        
        if row['is_artist_deep_dive'] == 1:
            intent_candidates.append(WEIGHT_DEEP_DIVE * scale)
        
        if row['is_context_switch'] == 1:
            # 맥락 전환은 직전 스트릭 길이가 아닌, 현재 곡이 새 스트릭의 첫 곡이므로
            # 의도적 전환 자체에 고정 점수 부여
            intent_candidates.append(WEIGHT_SWITCH)
        
        # 중복 방지: 가장 높은 행동 신호만 적용
        intent_bonus = max(intent_candidates) if intent_candidates else 0.0
        
        total = min(1.0, search_bonus + intent_bonus)
        scores.append(round(total, 2))
    
    df['proactive_score'] = scores
    
    return df


# ===================================================================
# 5. 메인 실행
# ===================================================================
def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 경로 설정
    search_html_path = os.path.join(
        BASE_DIR, 'Takeout', 'YouTube 및 YouTube Music', '시청 기록', '검색 기록.html'
    )
    features_csv_path = os.path.join(
        BASE_DIR, 'Takeout', 'YouTube 및 YouTube Music', '시청 기록', 'ytm_history_features.csv'
    )
    output_csv_path = os.path.join(
        BASE_DIR, 'Takeout', 'YouTube 및 YouTube Music', '시청 기록', 'ytm_history_features.csv'
    )
    
    if not os.path.exists(search_html_path):
        print(f"❌ 검색 기록 파일 없음: {search_html_path}")
        return
    if not os.path.exists(features_csv_path):
        print(f"❌ 피처 CSV 없음: {features_csv_path}")
        return
    
    print("=" * 60)
    print("🔍 능동적 재생 판별 엔진 (검색 기록 + 간접 신호)")
    print("=" * 60)
    
    # Step 1: 검색 기록 파싱
    search_records = parse_search_history(search_html_path)
    
    print("\n--- 최근 검색 10건 미리보기 ---")
    for s in search_records[:10]:
        print(f"  [{s['timestamp'].strftime('%Y-%m-%d %H:%M')}] \"{s['search_query']}\"")
    
    # Step 2: 재생 기록 로드
    print(f"\n[2/5] 재생 기록 로드 중...")
    play_df = pd.read_csv(features_csv_path, encoding='utf-8-sig')
    # 이전 실행의 컬럼이 남아있으면 제거
    for col in ['is_proactive', 'proactive_query', 'match_level', 
                'is_session_start', 'is_artist_deep_dive', 'is_context_switch', 
                'streak_length', 'proactive_score']:
        if col in play_df.columns:
            play_df = play_df.drop(columns=[col])
    print(f"  → {len(play_df)}건 로드 완료")
    
    # Step 3: 검색-재생 매칭
    matcher = SearchPlayMatcher(time_window_minutes=10)
    result_df = matcher.match_searches_to_plays(search_records, play_df)
    
    # Step 4: 간접 신호 감지
    result_df = detect_indirect_proactive_signals(result_df)
    
    # Step 5: 복합 능동성 점수 산출
    result_df = compute_proactive_score(result_df)
    
    # ======== 리포트 ========
    print(f"\n{'=' * 60}")
    print(f"📊 복합 능동성 분석 결과")
    print(f"{'=' * 60}")
    
    total = len(result_df)
    searched = result_df['is_proactive'].sum()
    session_start = result_df['is_session_start'].sum()
    deep_dive = result_df['is_artist_deep_dive'].sum()
    ctx_switch = result_df['is_context_switch'].sum()
    
    print(f"  전체 재생: {total}건")
    print(f"  🔍 검색 후 재생: {searched}건 ({searched/total*100:.1f}%)")
    print(f"  ▶️ 세션 시작곡: {session_start}건 ({session_start/total*100:.1f}%)")
    print(f"  🎵 아티스트 연속(4곡+): {deep_dive}건 ({deep_dive/total*100:.1f}%)")
    print(f"  🔀 맥락 전환(4곡+ 후): {ctx_switch}건 ({ctx_switch/total*100:.1f}%)")
    
    # 능동성 점수 분포
    print(f"\n--- 능동성 점수(proactive_score) 분포 ---")
    bins = [0, 0.01, 0.2, 0.3, 0.5, 0.7, 1.01]
    labels = ['0.0 (완전 수동)', '~0.2 (약한 신호)', '~0.3 (세션 시작)', 
              '~0.5 (검색 or 복합)', '~0.7 (강한 의도)', '0.8+ (확실한 능동)']
    result_df['score_bin'] = pd.cut(result_df['proactive_score'], bins=bins, labels=labels, right=False)
    for label in labels:
        count = len(result_df[result_df['score_bin'] == label])
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"  {label:<25} {count:>5}건 ({pct:>5.1f}%) |{bar}")
    result_df = result_df.drop(columns=['score_bin'])
    
    # 아티스트별 평균 능동성 점수 TOP 15 (핵심 분석!)
    print(f"\n--- 🎯 아티스트별 평균 능동성 점수 TOP 15 (최소 20회 재생) ---")
    artist_proactive = result_df.groupby('artist').agg(
        play_count=('proactive_score', 'count'),
        avg_score=('proactive_score', 'mean'),
        search_count=('is_proactive', 'sum'),
        session_start_count=('is_session_start', 'sum'),
        deep_dive_count=('is_artist_deep_dive', 'sum'),
        ctx_switch_count=('is_context_switch', 'sum'),
    )
    artist_proactive = artist_proactive[artist_proactive['play_count'] >= 20]
    artist_proactive = artist_proactive.sort_values('avg_score', ascending=False).head(15)
    
    print(f"  {'아티스트':<28} {'재생':>5} {'능동':>6} {'검색':>5} {'세션':>4} {'연속':>4} {'전환':>4}")
    print(f"  {'-'*72}")
    for artist, row in artist_proactive.iterrows():
        a = artist[:26]
        bar = '█' * int(row['avg_score'] * 20)
        print(f"  {a:<28} {row['play_count']:>5.0f} {row['avg_score']:>5.2f}  {row['search_count']:>5.0f} {row['session_start_count']:>4.0f} {row['deep_dive_count']:>4.0f} {row['ctx_switch_count']:>4.0f}  |{bar}")
    
    # 저장
    if 'play_timestamp' in result_df.columns:
        result_df = result_df.drop(columns=['play_timestamp'])
    
    result_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장 완료: {output_csv_path}")
    print(f"   → 추가된 컬럼: is_proactive, proactive_query, match_level,")
    print(f"     is_session_start, is_artist_deep_dive, is_context_switch,")
    print(f"     streak_length, proactive_score")


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
