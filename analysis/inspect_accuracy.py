from lifecycle_recommender import ArtistTierClassifier, SongTemperatureTracker, AsymmetricFlowDetector, PlaylistMixer
import pandas as pd
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

def detailed_inspection(csv_path, user_name):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    # 1. Tier 분류
    tier_classifier = ArtistTierClassifier(df)
    tier_map = tier_classifier.classify_tiers()
    
    # 2. 온도 판별
    temp_tracker = SongTemperatureTracker(df, tier_map)
    song_temps = temp_tracker.classify_all_songs()
    
    # 3. 비대칭 흐름 감지
    flow_detector = AsymmetricFlowDetector(df, tier_map)
    growth = flow_detector.detect_growth_bottom_up()
    
    # 4. 플레이리스트 생성 (50곡)
    mixer = PlaylistMixer(song_temps, growth)
    playlist = mixer.generate_playlist(total_songs=50, preset='default')
    
    print(f"\n{'#'*60}")
    print(f"# 🎧 {user_name} — 초정밀 추천 리스트 (50곡)")
    print(f"{'#'*60}\n")
    
    mixer.display_playlist(playlist, 'default')
    
    # 5. 특정 아티스트 집중 분석 (잔나비 vs 찰리푸스)
    print("\n" + "="*60)
    print("🔍 특정 아티스트 곡별 상태 집중 분석")
    print("="*60)
    
    target_artists = ['JANNABI - Topic', 'Charlie Puth - Topic', 'The Black Skirts - Topic']
    
    song_data = pd.DataFrame(song_temps.values())
    
    for artist in target_artists:
        print(f"\n📌 아티스트: {artist}")
        artist_songs = song_data[song_data['artist'] == artist].sort_values('total_plays', ascending=False)
        
        if artist_songs.empty:
            print("  데이터 없음 (이름을 확인하세요)")
            continue
            
        print(f"  {'온도':<6} | {'재생':<4} | {'마지막':<6} | {'곡명'}")
        print("  " + "-"*60)
        
        for _, row in artist_songs.head(15).iterrows():
            temp_icon = {'Hot': '🔥', 'Warm': '🟡', 'Cool': '🧊', 'Frozen': '❄️', 'Nostalgia': '🕰️'}.get(row['temperature'], '❓')
            print(f"  {temp_icon} {row['temperature']:<4} | {row['total_plays']:>4} | {row['days_since_last']:>6} | {row['title'][:40]}")

if __name__ == '__main__':
    csv_path = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
    detailed_inspection(csv_path, 'user')
