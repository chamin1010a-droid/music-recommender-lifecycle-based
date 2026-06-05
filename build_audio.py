"""
[오디오 특성 수집 스크립트]
2817곡의 YouTube 오디오를 다운로드하고 librosa로 음향 특성을 추출합니다.
예상 소요: 약 5시간+ (백그라운드 실행 권장)

캐시 자동 저장: 50곡마다 data/caches/audio_features_cache.json에 저장
→ 중단해도 이어서 진행 가능
"""
import subprocess, os
# FFmpeg PATH 갱신 (winget 설치 직후 셸에서 인식 안 될 수 있음)
ffmpeg_paths = [
    r'C:\ProgramData\chocolatey\bin',
    r'C:\Users\user\AppData\Local\Microsoft\WinGet\Links',
]
for p in ffmpeg_paths:
    if os.path.isdir(p) and p not in os.environ.get('PATH', ''):
        os.environ['PATH'] = p + ';' + os.environ.get('PATH', '')
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from audio_features_engine import AudioFeaturesEngine

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
print(f"오디오 다운로드 및 특성 추출 시작...\n")

engine = AudioFeaturesEngine()
engine.build_features(song_dict, batch_save_interval=50)

print("\n=== 오디오 특성 수집 완료 ===")
