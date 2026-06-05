from lifecycle_recommender import run_pipeline
import sys
sys.stdout.reconfigure(encoding='utf-8')

csv_path = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube 기\시청 기록\ytm_history_features.csv'
# 오타 수정
csv_path = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'

result = run_pipeline(csv_path, 'user(사용자)', playlist_size=20, preset='default')

song_temps = result['temp_tracker'].song_temps

# DAY6 곡들 중에서 'downgraded_by_skip' 이 True인 곡을 찾아보자
print("\n\n=== 스킵률에 의한 온도 강등 (DAY6 곡들) ===")
hit = False
for sid, info in song_temps.items():
    artist = info.get('artist', '')
    if isinstance(artist, float):
        continue
    if "DAY6" in artist and info.get('downgraded_by_skip'):
        hit = True
        print(f"곡명: {info['title']}, 현재 온도: {info['temperature']}, "
              f"스킵률: {info.get('skip_rate', 0):.2f}, Z-Score: {info.get('skip_z_score', 0):.2f}")

if not hit:
    print("DAY6 곡 중 스킵률로 강등된 곡이 없습니다.")
    day6_page = [info for sid, info in song_temps.items() if "한 페이지가 될 수 있게" in info['title']]
    if day6_page:
        info = day6_page[0]
        print(f"\n'한 페이지가 될 수 있게'의 현재 상태: 온도={info['temperature']}, 스킵률={info.get('skip_rate', 0):.2f}, Z-Score={info.get('skip_z_score', 0):.2f}")
    
