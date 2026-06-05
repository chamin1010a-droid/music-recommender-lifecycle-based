import pandas as pd
import numpy as np
import os
import sys

# Set stdout to utf-8
sys.stdout.reconfigure(encoding='utf-8')

csv_path = r"c:\Users\user\Desktop\데이터분석\음악 프로젝트\유튜브 뮤직 로그들\친구C\친구C_features.csv"
df = pd.read_csv(csv_path, encoding='utf-8-sig')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Next-listen interval
df = df.sort_values(['song_id', 'timestamp'])
df['interval_h'] = (df.groupby('song_id')['timestamp'].shift(-1) - df['timestamp']).dt.total_seconds() / 3600

# User activity (weekly)
df['week'] = df['timestamp'].dt.to_period('W').apply(lambda r: r.start_time)
weekly_stats = df.groupby('week')['song_id'].count().rename('activity')
df = df.merge(weekly_stats, on='week', how='left')

target_songs = df['song_id'].value_counts()[df['song_id'].value_counts() >= 20].index

print(f"--- [친구C] 개별 곡 정밀 추적 (고반복 곡 {len(target_songs)}개) ---")

for song in target_songs:
    s_df = df[df['song_id'] == song].sort_values('timestamp').reset_index(drop=True)
    n = len(s_df)
    
    # Stages: Early (First 5), Middle (10-15), Late (Last 5)
    early = s_df.head(5)
    late = s_df.tail(5)
    
    sat_e, sat_l = early['satisfaction'].mean(), late['satisfaction'].mean()
    int_e, int_l = early['interval_h'].median(), late['interval_h'].median()
    act_e, act_l = early['activity'].mean(), late['activity'].mean()
    
    print(f"\n[{song}] (총 {n}회 재생)")
    print(f"  만족도: {sat_e:.2f} (초기) -> {sat_l:.2f} (최근)")
    print(f"  간격(중앙값): {int_e:.1f}h (초기) -> {int_l:.1f}h (최근)")
    print(f"  전체 활동량(주간): {act_e:.1f}곡 -> {act_l:.1f}곡")
    
    # Logic verification
    if sat_l < sat_e - 0.3:
        # Satisfaction dropped
        if act_l < act_e * 0.7:
             print("  >>> 결론: [전체 활동 감소] 곡 자체보다 사용자의 청취 활동 자체가 줄어든 시기입니다.")
        elif int_l > int_e * 1.5:
             print("  >>> 결론: [명확한 질림] 활동량은 유지되지만 이 곡의 청취 빈도와 만족도가 모두 감소했습니다.")
        else:
             print("  >>> 결론: [심리적 질림] 듣긴 듣지만 스킵이 늘어나는 권태 구간입니다.")
    elif sat_l >= sat_e - 0.2:
        print("  >>> 결론: [여전한 애정] 반복 재생에도 불구하고 만족도와 관심도가 유지되고 있습니다.")

