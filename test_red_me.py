import os
import sys, os
sys.path.append(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\core')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from lyrics_engine import LyricsEngine

csv_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
df = pd.read_csv(csv_p, encoding='utf-8-sig')

# 검정치마 곡 목록
bs = df[df['artist'].str.contains('Black Skirts', case=False, na=False)][['song_id','title','artist']].drop_duplicates('song_id')

engine = LyricsEngine(genius_token=os.environ.get("GENIUS_TOKEN", ""))
engine._build_matrix()

# 시드 찾기
seed_id = None
for _, row in bs.iterrows():
    if row['title'] == 'Holiday':
        seed_id = row['song_id']
        print(f"시드: Holiday (빨간 나를) | {row['song_id'][:50]}")
        break

if not seed_id:
    print("빨간 나를 못 찾음!"); exit()

# 가사 미리보기
lyr = engine.lyrics_cache.get(seed_id, '')
if lyr:
    print(f"\n🎤 가사 앞 200자:\n{lyr[:200]}\n")
else:
    print("⚠️ 가사 없음!\n")

results = []
for _, row in bs.iterrows():
    sid = row['song_id']
    if sid == seed_id:
        continue
    sim = engine.calculate_similarity(seed_id, sid)
    has_lyrics = engine.lyrics_cache.get(sid) is not None
    results.append({
        'title': row['title'],
        'sim': sim,
        'has_lyrics': '✅' if has_lyrics else '❌',
    })

results.sort(key=lambda x: x['sim'] if x['sim'] is not None else -1, reverse=True)

print(f'| # | 곡명 | 가사유사도 | 가사? |')
print(f'|--:|:---|---:|:---:|')
for i, r in enumerate(results, 1):
    t = r['title'][:45]
    s = f"{r['sim']:.3f}" if r['sim'] is not None else '-'
    print(f"| {i} | {t} | {s} | {r['has_lyrics']} |")
