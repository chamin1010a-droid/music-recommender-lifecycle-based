import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'core'))

from lifecycle_recommender import run_pipeline

target_csv = os.path.join(
    BASE_DIR, 'Takeout', 'YouTube 및 YouTube Music', 
    '시청 기록', 'ytm_history_features.csv'
)
metadata_path = os.path.join(BASE_DIR, 'data', 'caches', 'ytm_metadata_cache.csv')
if not os.path.exists(metadata_path):
    metadata_path = None

def show_scores(result, target_artists):
    scorer = result['scorer']
    for song_id, info in scorer.song_scores.items():
        for t_artist in target_artists:
            artist_name = info.get('artist', '')
            if isinstance(artist_name, str) and t_artist.lower() in artist_name.lower():
                print(f"[{info['artist'][:15]}] {info['title'][:30]}")
                print(f"  Affinity: {info.get('affinity', 0):.2f} | Momentum: {info.get('momentum', 0):.2f}")
                break

print("=== 1. 기본 실행 (시드 없음) ===")
result1 = run_pipeline(
    csv_path=target_csv,
    user_name='user_기본',
    playlist_size=20,
    preset='default',
    metadata_path=metadata_path,
    user_birth_year=1998
)

print("\n--- Jay Park / KARA 점수 검증 ---")
show_scores(result1, ['Jay Park', 'KARA'])

print("\n\n=== 2. KARA 시드 기반 실행 ===")
kara_seed = [{'artist': 'KARA - Topic', 'title': 'Pretty Girl'}]
result2 = run_pipeline(
    csv_path=target_csv,
    user_name='user_KARA_Seed',
    playlist_size=20,
    preset='default',
    metadata_path=metadata_path,
    user_birth_year=1998,
    seed_tracks=kara_seed
)
