import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Set stdout to utf-8
sys.stdout.reconfigure(encoding='utf-8')

# Set Korean Font
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def analyze_individual_songs(friend_name, csv_file, min_plays=15):
    print(f"\n--- [{friend_name}] 개별 곡 정밀 분석 ---")
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    
    # Filter for target songs
    song_counts = df['song_id'].value_counts()
    high_play_songs = song_counts[song_counts >= min_plays].index[:10] # Top 10
    
    print(f"재생 횟수 {min_plays}회 이상인 상위 10곡을 확인합니다.")
    
    results = []
    
    for song in high_play_songs:
        song_df = df[df['song_id'] == song].sort_values('timestamp').reset_index(drop=True)
        # Group by buckets of 3 plays to smooth the noise
        song_df['play_bucket'] = song_df.index // 3
        bucket_satisfaction = song_df.groupby('play_bucket')['satisfaction_score'].mean()
        
        # Journey description
        start_score = bucket_satisfaction.iloc[0]
        peak_score = bucket_satisfaction.max()
        peak_idx = bucket_satisfaction.idxmax()
        end_score = bucket_satisfaction.iloc[-1]
        
        trend = "상승형" if end_score > start_score else "하강형"
        if peak_idx > 0 and peak_idx < len(bucket_satisfaction) - 1:
            trend = "골디락스형 (정점 후 하락)"
            
        results.append({
            'song': song,
            'total_plays': len(song_df),
            'start_score': round(start_score, 2),
            'peak_score': round(peak_score, 2),
            'end_score': round(end_score, 2),
            'trend': trend
        })
        
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    return res_df

# Run for both
base_dir = r"c:\Users\user\Desktop\데이터분석\음악 프로젝트\유튜브 뮤직 로그들"

seo_csv = os.path.join(base_dir, "친구D", "친구D_features.csv")
bum_csv = os.path.join(base_dir, "친구B", "친구B_features.csv")

if os.path.exists(seo_csv):
    analyze_individual_songs("친구D", seo_csv)

if os.path.exists(bum_csv):
    analyze_individual_songs("친구B", bum_csv)
