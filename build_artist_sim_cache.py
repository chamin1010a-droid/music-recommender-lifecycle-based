import os
"""
🌙 밤에 돌릴 Last.fm 아티스트 유사도 캐싱 스크립트

전체 아티스트에 대해 Last.fm similar artists를 조회하여 캐시에 저장.
캐시된 후에는 추천 엔진에서 API 호출 없이 즉시 사용 가능.

예상 시간: ~150 아티스트 × 0.15초/호출 = ~25초
(이미 캐시된 아티스트는 스킵)

사용법: python build_artist_sim_cache.py
"""

import sys, os, io, time
sys.path.append(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\core')
sys.stdout.reconfigure(encoding='utf-8')

from lifecycle_recommender import run_pipeline
from lastfm_client import LastFMClient

LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY", "")

csv_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
meta_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\data\caches\ytm_metadata_cache.csv'

# 1. 파이프라인 로드 (skip_external로 빠르게)
print("📦 파이프라인 로딩...", flush=True)
old = sys.stdout
sys.stdout = io.TextIOWrapper(open(os.devnull, 'wb'), encoding='utf-8')
r = run_pipeline(csv_path=csv_p, user_name='cache', playlist_size=5, preset='default',
                 metadata_path=meta_p, user_birth_year=1998, skip_external=True)
sys.stdout = old
sys.stdout.reconfigure(encoding='utf-8')
print("✅ 파이프라인 로드 완료!")

mixer = r['mixer']
song_temps = mixer.song_temps

# 2. 전체 아티스트 추출
from collections import Counter
artist_plays = Counter()
for info in song_temps.values():
    artist = info.get('artist', '').replace(' - Topic', '').strip()
    if artist:
        artist_plays[artist] += info.get('total_plays', 0)

all_artists = [a for a, _ in artist_plays.most_common()]
print(f"\n🎤 전체 아티스트: {len(all_artists)}명")

# 3. Last.fm 캐시 파일 확인
cache_file = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\data\caches\lastfm_artist_sim_cache.json'
os.makedirs(os.path.dirname(cache_file), exist_ok=True)

lastfm = LastFMClient(api_key=LASTFM_API_KEY, cache_file=cache_file)

# 이미 캐시된 아티스트 확인
cached = [a for a in all_artists if f"similar_artists||{a}" in lastfm.cache]
to_fetch = [a for a in all_artists if f"similar_artists||{a}" not in lastfm.cache]

print(f"  이미 캐시됨: {len(cached)}명")
print(f"  새로 조회할 아티스트: {len(to_fetch)}명")
print(f"  예상 시간: {len(to_fetch) * 0.15:.0f}초 ({len(to_fetch) * 0.15 / 60:.1f}분)")

# 4. 캐싱 실행
print(f"\n🔍 Last.fm 유사 아티스트 캐싱 시작...\n")

start = time.time()
success = 0
fail = 0
empty = 0

for i, artist in enumerate(to_fetch):
    try:
        results = lastfm.get_similar_artists(artist, limit=50)
        if results:
            success += 1
            top3 = [r['name'] for r in results[:3]]
            print(f"  [{i+1}/{len(to_fetch)}] ✅ {artist:30s} → {top3}")
        else:
            empty += 1
            print(f"  [{i+1}/{len(to_fetch)}] ⚪ {artist:30s} → (결과 없음)")
    except Exception as e:
        fail += 1
        print(f"  [{i+1}/{len(to_fetch)}] ❌ {artist:30s} → {e}")
    
    # 진행률 (50개마다)
    if (i + 1) % 50 == 0:
        elapsed = time.time() - start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        remaining = (len(to_fetch) - i - 1) / rate if rate > 0 else 0
        print(f"\n  --- 진행: {i+1}/{len(to_fetch)} | "
              f"성공 {success} / 빈결과 {empty} / 실패 {fail} | "
              f"남은 시간: {remaining:.0f}초 ---\n")

elapsed = time.time() - start

# 5. 결과 요약
print(f"\n{'='*60}")
print(f"🌙 Last.fm 아티스트 유사도 캐싱 완료!")
print(f"{'='*60}")
print(f"  전체 아티스트: {len(all_artists)}명")
print(f"  신규 캐시: {success + empty}명 (성공 {success} / 빈결과 {empty} / 실패 {fail})")
print(f"  이전 캐시: {len(cached)}명")
print(f"  총 캐시: {len(cached) + success + empty}명")
print(f"  소요 시간: {elapsed:.1f}초")
print(f"  캐시 파일: {cache_file}")
print(f"\n  다음에 추천 엔진 실행 시 skip_external=False로 실행하면")
print(f"  이 캐시를 자동으로 사용합니다!")
