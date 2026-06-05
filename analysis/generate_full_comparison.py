"""
전곡 분류 비교 리포트 생성기
- 현재 방식: lifecycle_recommender.py의 온도(Temperature) 분류
- 예전 방식: Zone 1~4 (전반전 vs 후반전 스킵률 비교) 분류
두 방식 모두 전곡을 CSV로 출력하여 사용자가 직접 비교 검증할 수 있게 함.
"""
import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')
from lifecycle_recommender import run_pipeline

# =============================================
# 1. 현재 방식 (Temperature) 전곡 분류
# =============================================
history_csv = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
meta_csv = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\ytm_metadata_cache.csv'

result = run_pipeline(
    csv_path=history_csv,
    user_name='user',
    playlist_size=15,
    preset='default',
    metadata_path=meta_csv,
    user_birth_year=1998
)
temps = result['temp_tracker'].song_temps

# =============================================
# 2. 예전 방식 (Zone 1~4) 전곡 분류
# =============================================
df = pd.read_csv(history_csv, encoding='utf-8-sig')
df['timestamp'] = pd.to_datetime(df['timestamp'])

zone_results = {}
for song_id, group in df.groupby('song_id'):
    group = group.sort_values('timestamp')
    title = group['title'].iloc[0]
    artist = group['artist'].iloc[0]
    total_plays = len(group)
    
    if total_plays < 3:
        # 재생 3회 미만은 판단 불가
        zone = 'Zone N/A (재생 부족)'
        first_half_skip = np.nan
        second_half_skip = np.nan
        first_3_skip = np.nan
        overall_skip = np.nan
    else:
        overall_skip = group['is_skipped'].mean()
        half = len(group) // 2
        first_half = group.iloc[:half]
        second_half = group.iloc[half:]
        first_half_skip = first_half['is_skipped'].mean()
        second_half_skip = second_half['is_skipped'].mean()
        first_3_skip = group.iloc[:min(3, len(group))]['is_skipped'].mean()
        
        # Zone 분류 로직
        if first_3_skip <= 0.33 and overall_skip <= 0.3:
            zone = 'Zone 1 (처음부터 좋아함)'
        elif first_half_skip > second_half_skip + 0.15:
            zone = 'Zone 2 (듣다보니 좋아짐)'
        elif second_half_skip > first_half_skip + 0.15:
            zone = 'Zone 3 (좋다가 질림)'
        else:
            zone = 'Zone 4 (그냥저냥)'
    
    zone_results[song_id] = {
        'zone': zone,
        'first_half_skip': round(first_half_skip, 2) if not pd.isna(first_half_skip) else None,
        'second_half_skip': round(second_half_skip, 2) if not pd.isna(second_half_skip) else None,
        'overall_skip_zone': round(overall_skip, 2) if not pd.isna(overall_skip) else None
    }

# =============================================
# 3. 두 분류 결합 -> CSV 출력
# =============================================
rows = []
for song_id, info in temps.items():
    zone_info = zone_results.get(song_id, {})
    rows.append({
        '아티스트': info['artist'],
        '곡명': info['title'],
        '아티스트Tier': info['tier'],
        '총재생수': info['total_plays'],
        '스킵률': f"{info['skip_rate']*100:.0f}%",
        '마지막재생(일전)': info['days_since_last'],
        '추세기울기': info['trend_slope'],
        '최근14일재생': info['recent_14d_plays'],
        '장르': info.get('genre', ''),
        '청취시간대': info.get('peak_time_of_day', ''),
        '스킵Z점수': info['skip_z_score'],
        '스킵강등여부': '⚠️강등' if info['downgraded_by_skip'] else '',
        # --- 현재 방식 ---
        '[현재] 온도등급': info['temperature'],
        # --- 예전 방식 ---
        '[예전] Zone분류': zone_info.get('zone', 'N/A'),
        '[예전] 전반스킵률': zone_info.get('first_half_skip', ''),
        '[예전] 후반스킵률': zone_info.get('second_half_skip', ''),
    })

out_df = pd.DataFrame(rows)

# 정렬: 아티스트Tier(S>A>B) -> 총재생수 내림차순
tier_order = {'S': 0, 'A': 1, 'B': 2}
out_df['_tier_sort'] = out_df['아티스트Tier'].map(tier_order)
out_df = out_df.sort_values(['_tier_sort', '총재생수'], ascending=[True, False])
out_df = out_df.drop(columns=['_tier_sort'])

output_path = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\full_classification_comparison.csv'
out_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"✅ 전곡 분류 비교 CSV 생성 완료: {output_path}")
print(f"   총 {len(out_df)}곡")
print()

# 요약 통계
print("="*60)
print("[현재 방식] 온도(Temperature) 등급별 곡 수")
print("="*60)
for temp in ['Rising', 'Steady', 'Warm', 'Cool', 'Frozen', 'Nostalgia']:
    count = len(out_df[out_df['[현재] 온도등급'] == temp])
    print(f"  {temp:>12}: {count:>5}곡")

print()
print("="*60)
print("[예전 방식] Zone 분류별 곡 수")
print("="*60)
for zone_label in sorted(out_df['[예전] Zone분류'].unique()):
    count = len(out_df[out_df['[예전] Zone분류'] == zone_label])
    print(f"  {zone_label}: {count:>5}곡")
