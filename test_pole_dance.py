import os
import sys, os
sys.path.append(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\core')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from lyrics_engine import LyricsEngine

csv_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
df = pd.read_csv(csv_p, encoding='utf-8-sig')
unique = df[['song_id','title','artist']].drop_duplicates('song_id')

print('--- 엔진 로딩 ---')
engine = LyricsEngine(genius_token=os.environ.get("GENIUS_TOKEN", ""))
engine._build_matrix()

seed_id = 'Pole Dance (봉춤을 추네) - JANNABI - Topic'

# 시드 가사 미리보기
seed_lyrics = engine.lyrics_cache.get(seed_id, '')
if seed_lyrics:
    print(f'\n🎤 봉춤을 추네 가사 (앞 150자):')
    print(seed_lyrics[:150])
    print('...\n')

results = []
for _, row in unique.iterrows():
    sid = row['song_id']
    if sid == seed_id:
        continue
    sim = engine.calculate_similarity(seed_id, sid)
    if sim is not None:
        results.append({
            'title': row['title'],
            'artist': row['artist'].replace(' - Topic',''),
            'lyrics_sim': sim,
        })

results.sort(key=lambda x: x['lyrics_sim'], reverse=True)

print(f'| # | 곡명 | 아티스트 | 가사유사도 |')
print(f'|--:|:---|:---|---:|')
for i, r in enumerate(results[:30], 1):
    t = r['title'][:40]
    a = r['artist'][:15]
    print(f"| {i} | {t} | {a} | {r['lyrics_sim']:.3f} |")

print(f'\n--- 하위 10곡 ---')
print(f'| # | 곡명 | 아티스트 | 가사유사도 |')
print(f'|--:|:---|:---|---:|')
for i, r in enumerate(results[-10:], len(results)-9):
    t = r['title'][:40]
    a = r['artist'][:15]
    print(f"| {i} | {t} | {a} | {r['lyrics_sim']:.3f} |")
