import os
"""
Last.fm 동명이인 오인 수정 스크립트

1. 라이브러리 아티스트 중 Last.fm에서 외국 동명이인으로 매칭된 케이스 찾기
   → 유사 아티스트에 우리 라이브러리 아티스트가 하나도 없으면 의심
2. 해당 아티스트를 한국어 이름으로 재검색
"""
import sys, os, io, json, time
sys.path.append(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\core')
sys.stdout.reconfigure(encoding='utf-8')
from lastfm_client import LastFMClient
from lifecycle_recommender import run_pipeline

LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY", "")
cache_file = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\data\caches\lastfm_artist_sim_cache.json'

# 파이프라인 로드 (아티스트 목록 얻기)
csv_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
meta_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\data\caches\ytm_metadata_cache.csv'
old = sys.stdout
sys.stdout = io.TextIOWrapper(open(os.devnull, 'wb'), encoding='utf-8')
r = run_pipeline(csv_path=csv_p, user_name='fix', playlist_size=5, preset='default',
                 metadata_path=meta_p, user_birth_year=1998, skip_external=True)
sys.stdout = old
sys.stdout.reconfigure(encoding='utf-8')

mixer = r['mixer']

# 전체 라이브러리 아티스트 (clean name)
lib_artists = set()
for info in mixer.song_temps.values():
    a = info.get('artist', '').replace(' - Topic', '').strip().lower()
    if a:
        lib_artists.add(a)

print(f"라이브러리 아티스트: {len(lib_artists)}명")

# Last.fm 캐시 로드
cache = json.load(open(cache_file, 'r', encoding='utf-8'))

# 동명이인 의심 탐지: 유사 아티스트 중 라이브러리에 있는 게 0개
suspect = []
for key, sims in cache.items():
    if not key.startswith('similar_artists||'):
        continue
    artist = key.replace('similar_artists||', '')
    
    if not sims:  # 빈 결과는 스킵
        continue
    
    # 유사 아티스트 중 라이브러리에 있는 수
    overlap = 0
    for s in sims:
        if s['name'].lower() in lib_artists:
            overlap += 1
    
    # 재생 횟수 확인 (중요한 아티스트만)
    total_plays = 0
    for info in mixer.song_temps.values():
        if info.get('artist', '').replace(' - Topic', '').strip() == artist:
            total_plays += info.get('total_plays', 0)
    
    if overlap == 0 and total_plays >= 5:
        top3 = [s['name'] for s in sims[:3]]
        suspect.append((artist, total_plays, top3))

suspect.sort(key=lambda x: x[1], reverse=True)

print(f"\n🚨 동명이인 의심 아티스트 ({len(suspect)}명):")
print(f"   (유사 아티스트에 라이브러리 곡이 0개 & 재생 5회 이상)")
print(f"\n{'아티스트':25s} | {'재생':>5s} | Last.fm 유사 TOP 3")
print("-" * 80)
for artist, plays, top3 in suspect:
    print(f"{artist:25s} | {plays:5d} | {top3}")

# 한국 아티스트 재검색 시도
print(f"\n\n🔧 한국어 이름으로 재검색 시도...")
lastfm = LastFMClient(api_key=LASTFM_API_KEY, cache_file=cache_file)

# 알려진 한국어 이름 매핑
korean_names = {
    'December': '디셈버',
    'Toy': '토이 (유희열)',
    'Flower': '플라워',
    'E.R.U.': '이루',
    'ISU': '이수',
    'Postmen': '포스트맨',
    'Sonnet': '소넷',
    'Turbo': '터보',
    'COOL': '쿨',
    'HOT': '에이치오티',
    'Crush': '크러쉬',
    'MAX': '엠씨더맥스',
}

fixed = 0
for artist, plays, old_top3 in suspect:
    korean = korean_names.get(artist)
    if korean:
        # 한국어 이름으로 재검색
        cache_key_old = f"similar_artists||{artist}"
        cache_key_kr = f"similar_artists||{korean}"
        
        # 이미 한국어로 검색한 적 있으면 그 결과 사용
        if cache_key_kr in lastfm.cache:
            new_sims = lastfm.cache[cache_key_kr]
        else:
            new_sims = lastfm.get_similar_artists(korean, limit=50)
        
        if new_sims:
            overlap = sum(1 for s in new_sims if s['name'].lower() in lib_artists)
            new_top3 = [s['name'] for s in new_sims[:3]]
            print(f"  ✅ {artist} → {korean}: {new_top3} (라이브러리 겹침: {overlap})")
            
            # 캐시 덮어쓰기
            lastfm.cache[cache_key_old] = new_sims
            fixed += 1
        else:
            print(f"  ⚪ {artist} → {korean}: 결과 없음")

lastfm._save_cache()
print(f"\n✅ {fixed}명 수정 완료! 캐시 저장됨.")
