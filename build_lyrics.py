import os
"""
[가사 수집 + 임베딩 스크립트]
Genius API로 2817곡 가사를 수집하고, multilingual 모델로 임베딩합니다.
예상 소요: 약 30~40분 (API rate limit 의존)

캐시 자동 저장: 50곡마다 data/caches/lyrics_cache.json에 저장
→ 중단해도 이어서 진행 가능
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from lyrics_engine import LyricsEngine

GENIUS_TOKEN = os.environ.get("GENIUS_TOKEN", "")

base_dir = r"c:\Users\user\Desktop\데이터분석\음악 프로젝트"
history_csv = os.path.join(base_dir, "Takeout", "YouTube 및 YouTube Music", "시청 기록", "ytm_history_features.csv")

print(f"데이터 파일: {history_csv}")
df = pd.read_csv(history_csv, encoding='utf-8-sig')
unique_songs = df[['song_id', 'artist', 'title']].drop_duplicates('song_id')

song_dict = {}
for _, row in unique_songs.iterrows():
    song_dict[row['song_id']] = {
        'artist': row['artist'],
        'title': row['title']
    }

print(f"총 고유 곡 수: {len(song_dict)}곡")
print(f"가사 수집 및 임베딩 분석 시작...\n")

engine = LyricsEngine(genius_token=GENIUS_TOKEN)
engine.build_embeddings(song_dict, batch_save_interval=50)

print("\n=== 가사 수집 + 임베딩 완료 ===")
