from lifecycle_recommender import run_pipeline
import sys
sys.stdout.reconfigure(encoding='utf-8')

csv_path = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
result = run_pipeline(csv_path, 'user(사용자)', playlist_size=20, preset='default')

# 오아시스 Tier 확인
print("\n\n=== 오아시스 Tier 결과 확인 ===")
tier_map = result['tier_classifier'].tier_map
for k, v in tier_map.items():
    if 'oasis' in k.lower() or 'Oasis' in k:
        print(f"  {k}: Tier {v}")

stats = result['tier_classifier'].artist_stats
oasis_row = stats[stats['artist'].str.contains('Oasis', case=False)]
print()
print(oasis_row[['artist', 'unique_songs', 'total_plays', 'recent_plays_30d', 'tier']].to_string())
