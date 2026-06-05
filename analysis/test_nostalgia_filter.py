from lifecycle_recommender import run_pipeline
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os

base = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록'
history_csv = os.path.join(base, 'ytm_history_features.csv')
metadata_csv = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\ytm_metadata_cache.csv'

# user님의 데이터로 실행
result = run_pipeline(
    csv_path=history_csv,
    user_name='user',
    playlist_size=20,
    preset='nostalgia',
    metadata_path=metadata_csv,
    user_birth_year=1998
)

temp_tracker = result['temp_tracker']
nostalgia_songs = [s for s in temp_tracker.song_temps.values() if s['temperature'] == 'Nostalgia']
frozen_songs = [s for s in temp_tracker.song_temps.values() if s['temperature'] == 'Frozen']

print("\n\n=== 🕰️ Nostalgia (향수) 판별 결과 ===")
for s in sorted(nostalgia_songs, key=lambda x: x['total_plays'], reverse=True)[:10]:
    meta = temp_tracker.metadata.get(s['song_id'], {})
    year_str = f"발매: {int(meta['release_year'])}" if meta.get('release_year') else "발매미상"
    print(f"[{s['tier']}] {s['title'][:30]:<32} - 재생 {s['total_plays']:>3}회 | {year_str}")

print(f"\n총 Nostalgia 곡 수: {len(nostalgia_songs)}")
print(f"총 Frozen 곡 수: {len(frozen_songs)}")

