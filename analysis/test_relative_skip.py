import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

def test_relative_skip_rate(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    # 아티스트 정규화 간이 구현
    df['artist'] = df['artist'].apply(lambda x: "DAY6 - Topic" if "DAY6" in str(x) else x)
    
    # 아티스트별 통계 (10곡 이상 재생된 아티스트만 대상)
    artist_skip_stats = {}
    
    for artist, group in df.groupby('artist'):
        if len(group) < 10:
            continue
            
        # 곡별 스킵 통계
        song_stats = group.groupby('song_id').agg(
            plays=('timestamp', 'count'),
            skips=('is_skipped', 'sum'),
            title=('title', 'first')
        )
        song_stats['skip_rate'] = song_stats['skips'] / song_stats['plays']
        
        # 3회 이상 재생된 곡만 대상으로 아티스트 평균 스킵률 계산
        valid_songs = song_stats[song_stats['plays'] >= 3]
        if len(valid_songs) < 3:
            continue
            
        mean_skip = valid_songs['skip_rate'].mean()
        std_skip = valid_songs['skip_rate'].std()
        if pd.isna(std_skip) or std_skip == 0:
            std_skip = 0.001
            
        artist_skip_stats[artist] = {
            'mean': mean_skip,
            'std': std_skip,
            'songs': song_stats
        }
        
        # DAY6만 출력해보자
        if "DAY6" in artist:
            print(f"=== {artist} ===")
            print(f"아티스트 평균 스킵률: {mean_skip:.3f}, 표준편차: {std_skip:.3f}")
            print("\n곡별 스킵 통계:")
            
            valid_songs['z_score'] = (valid_songs['skip_rate'] - mean_skip) / std_skip
            print(valid_songs[['title', 'plays', 'skip_rate', 'z_score']].sort_values('z_score', ascending=False).head(15).to_string())

if __name__ == '__main__':
    csv_path = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
    test_relative_skip_rate(csv_path)
