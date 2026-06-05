import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from lifecycle_recommender import run_pipeline
from lastfm_client import LastFMClient

CSV_PATH = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
META_PATH = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\ytm_metadata_cache.csv'
API_KEY = os.environ.get("LASTFM_API_KEY", "")

result = run_pipeline(CSV_PATH, 'user', 15, 'default', META_PATH, 1998)
song_temps = result['temp_tracker'].song_temps

lastfm = LastFMClient(api_key=API_KEY)

# 플레이리스트 시드곡 (잔나비 - All the Boys...) 의 태그
print("=== 시드곡 (잔나비) 태그 ===")
seed_tags = lastfm.get_combined_tags('JANNABI', 'All the Boys and Girls')
print(f"  {seed_tags}")

# 문제의 곡 3개 태그 비교
print("\n=== Discovery 후보곡들의 태그 ===")
test_songs = [
    ('Choi Yu Ree', 'Loneliness is'),   # 최유리 - 어울림
    ('M.C the MAX', '사랑의 시'),          # 엠씨더맥스 - 안어울림
    ('Maroon 5', 'Moves Like Jagger'),    # 마룬5 - 안어울림
    ('The Black Skirts', 'Breakfast'),    # 검정치마 - 어울림 (비교용)
    ('The Volunteers', 'Summer'),         # 볼런티어스 - 어울림 (비교용)
]

for artist, track in test_songs:
    tags = lastfm.get_combined_tags(artist, track)
    print(f"\n  [{artist} — {track}]")
    print(f"  태그: {tags[:10] if tags else '태그 없음!'}")

# 유사도 계산 과정 설명
print("\n" + "=" * 60)
print("📊 Discovery 슬롯의 유사도 계산 과정")
print("=" * 60)

# lifecycle_recommender의 _calculate_similarity 확인
sim_engine = result.get('similarity_engine')
if sim_engine:
    print(f"\n  태그 벡터 사전 크기: {len(sim_engine.tag_vectors)}곡")
    
    # 시드곡과 각 후보곡 간 유사도 비교
    for artist, track in test_songs:
        # song_temps에서 찾기
        found = None
        for sid, info in song_temps.items():
            a = str(info.get('artist', '')).replace(' - Topic', '').strip()
            t = str(info.get('title', ''))
            if artist.lower() in a.lower() and track.lower() in t.lower():
                found = (sid, info)
                break
        
        if found:
            sid, info = found
            print(f"\n  [{artist} — {track}]")
            print(f"    재생: {info.get('total_plays', 0)}회 | 온도: {info.get('temperature', '?')}")
            
            # 태그 벡터 존재 여부
            if sid in sim_engine.tag_vectors:
                print(f"    태그 벡터: ✅ 있음")
            else:
                print(f"    태그 벡터: ❌ 없음 → 유사도 0으로 계산됨!")
        else:
            print(f"\n  [{artist} — {track}] → 라이브러리에 없음")

print("\n" + "=" * 60)
print("💡 핵심 문제")
print("=" * 60)
print("""
  Discovery 슬롯 선정 과정:
  
  1. 성장 신호 아티스트의 적게 들은 곡 수집
  2. 부족하면 → 재생 1~3회 곡 중 무작위 보충
  3. 점수 = 태그유사도 + random(0~2) ← 이 랜덤이 문제!
  
  태그가 없는 곡은 유사도=0이지만
  random(0~2)에서 1.8 같은 높은 값이 나오면 선택됨!
  
  → 해결: 태그유사도가 일정 수준 이하면 아예 제외해야 함
""")
