import os
"""
기존 가사 캐시에서 오매칭된 가사를 찾아 제거하고,
해당 곡들의 임베딩도 삭제한 뒤 재수집합니다.
"""
import sys, os, json, re, time
sys.path.append(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\core')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from difflib import SequenceMatcher
from lyrics_engine import LyricsEngine

csv_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
df = pd.read_csv(csv_p, encoding='utf-8-sig')
unique = df[['song_id','title','artist']].drop_duplicates('song_id')

# song_id → artist 매핑
sid_to_artist = {}
for _, row in unique.iterrows():
    sid_to_artist[row['song_id']] = str(row['artist']).replace(' - Topic', '').replace('VEVO', '').strip()

# 엔진 로드 (캐시만)
engine = LyricsEngine(genius_token=os.environ.get("GENIUS_TOKEN", ""))

# 오매칭 검출: 캐시된 가사의 첫 줄에서 "XXX Lyrics" 패턴을 찾아
# Genius가 반환한 아티스트를 추정하고 우리 아티스트와 비교
# 아쉽게도 캐시에는 Genius 아티스트명이 저장되어 있지 않으므로,
# 대신 가사 내용이 비영어/비한국어인지 등 휴리스틱으로 검사

import unicodedata

def detect_suspicious(lyrics, expected_artist):
    """가사가 해당 아티스트 것이 아닌 것으로 의심되는지 검사"""
    if not lyrics:
        return False
    
    first_200 = lyrics[:200].lower()
    
    # 1. 아프리카/나이지리아 곡 패턴 (요루바어 등)
    yoruba_patterns = ['yin wo', 'fe da yan', 'dj jimmy', 'viktoh', 'lil kesh',
                       'shaku shaku', 'o ni di', 'wahala']
    for p in yoruba_patterns:
        if p in first_200:
            return True
    
    # 2. 스페인어 곡이 한국 아티스트에 매칭된 경우
    korean_artists = ['jannabi', 'the black skirts', 'hanroro', 'day6', 
                      'sg wannabe', 'december', 'monday kiz', 'car, the garden',
                      'nerd connection', 'hyukoh', 'nell', 'buzz', 'gummy',
                      'iu', 'bts', 'aespa', 'shinee', 'exo']
    
    is_korean_artist = any(k in expected_artist.lower() for k in korean_artists)
    
    if is_korean_artist:
        # 한국 아티스트인데 가사에 한글이 전혀 없고, 영어도 아닌 경우
        has_korean = any('\uac00' <= c <= '\ud7a3' for c in lyrics[:500])
        has_english = any('a' <= c.lower() <= 'z' for c in lyrics[:500])
        has_latin_special = any(unicodedata.category(c).startswith('L') and ord(c) > 127 
                                and not ('\uac00' <= c <= '\ud7a3') 
                                and not ('\u3040' <= c <= '\u30ff')  # 일본어 제외
                                for c in lyrics[:500])
        
        # 한글도 영어도 없으면 의심
        if not has_korean and not has_english and has_latin_special:
            return True
    
    return False

# 전수 검사
suspicious = []
for sid, lyrics in engine.lyrics_cache.items():
    if lyrics is None:
        continue
    expected_artist = sid_to_artist.get(sid, '')
    if detect_suspicious(lyrics, expected_artist):
        suspicious.append((sid, expected_artist, lyrics[:80]))

print(f"의심 곡 발견: {len(suspicious)}곡")
for sid, artist, preview in suspicious[:20]:
    print(f"  ❌ {sid[:50]} | {artist} | {preview[:60]}")

# 의심 곡 캐시에서 제거
removed = 0
for sid, _, _ in suspicious:
    engine.lyrics_cache[sid] = None  # 캐시를 None으로 (재수집 방지위해 유지)
    if sid in engine.embeddings:
        del engine.embeddings[sid]
        removed += 1

engine._save_lyrics_cache()
engine._save_embeddings_cache()
print(f"\n제거 완료: 캐시 {len(suspicious)}곡 무효화, 임베딩 {removed}곡 삭제")

# 재수집 시도 (검증 로직 포함된 새 fetch_lyrics 사용)
print(f"\n--- 오매칭 곡 재수집 시도 ({len(suspicious)}곡) ---")
recollected = 0
for sid, artist_name, _ in suspicious:
    # 캐시 강제 초기화 (재시도 가능하도록)
    if sid in engine.lyrics_cache:
        del engine.lyrics_cache[sid]
    
    info = unique[unique['song_id'] == sid]
    if len(info) == 0:
        continue
    row = info.iloc[0]
    lyrics = engine.fetch_lyrics(row['artist'], row['title'], song_id=sid)
    if lyrics:
        recollected += 1
        print(f"  ✅ 재수집 성공: {row['title'][:30]}")
    else:
        print(f"  ⚠️ 재수집 실패: {row['title'][:30]}")
    time.sleep(0.5)

engine._save_lyrics_cache()
print(f"\n재수집 결과: {recollected}/{len(suspicious)}곡 성공")
