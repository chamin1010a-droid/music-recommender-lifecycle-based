import sys
import pandas as pd
from lifecycle_recommender import run_pipeline

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    csv_path = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
    meta_path = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\ytm_metadata_cache.csv'

    result = run_pipeline(
        csv_path=csv_path,
        user_name='user',
        playlist_size=15,
        preset='default',
        metadata_path=meta_path,
        user_birth_year=1998
    )
    temps = result['temp_tracker'].song_temps

    for cat in ['Rising', 'Cool']:
        songs = [s for s in temps.values() if s['temperature'] == cat]
        # Sort by total plays (descending)
        songs.sort(key=lambda x: x['total_plays'], reverse=True)
        
        print(f"\n{'='*80}")
        print(f"🌡️ {cat} 등급 리포트 ({len(songs)}곡)")
        print(f"{'='*80}")
        print(f"{'아티스트':<20} | {'곡명':<35} | {'재생':>3} | {'스킵':>3} | {'최근':>3}")
        print("-" * 80)
        
        # Show top 50 or all if less
        for s in songs[:50]:
            artist = s['artist'][:20]
            title = s['title'][:35]
            plays = s['total_plays']
            skip = int(s['skip_rate'] * 100)
            last = s['days_since_last']
            print(f"{artist:<20} | {title:<35} | {plays:>4} | {skip:>4} | {last:>4}")
        
        if len(songs) > 50:
            print(f"...(외 {len(songs)-50}곡 더 있음)")

if __name__ == "__main__":
    main()
