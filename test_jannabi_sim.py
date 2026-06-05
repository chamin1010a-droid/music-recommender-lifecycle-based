import os
import sys, os
sys.path.append(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\core')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

# 엔진 로드
from lyrics_engine import LyricsEngine
from audio_features_engine import AudioFeaturesEngine
from multi_signal_engine import MultiSignalSimilarityEngine

csv_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
meta_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\data\caches\ytm_metadata_cache.csv'
df = pd.read_csv(csv_p, encoding='utf-8-sig')

# 잔나비 곡 목록
jannabi = df[df['artist'].str.contains('JANNABI', case=False, na=False)][['song_id','title','artist']].drop_duplicates('song_id')

print('--- 엔진 로딩 중 ---')
lyrics_engine = LyricsEngine(genius_token=os.environ.get("GENIUS_TOKEN", ""))
lyrics_engine._build_matrix()
audio_engine = AudioFeaturesEngine()
audio_engine._build_matrix()

multi = MultiSignalSimilarityEngine(
    tag_engine=None,
    lyrics_engine=lyrics_engine,
    audio_engine=audio_engine,
    metadata_path=meta_p
)

seed_id = 'bad dreams (나쁜 꿈) - JANNABI - Topic'
print(f'시드: bad dreams (나쁜 꿈)')
print()

results = []
for _, row in jannabi.iterrows():
    sid = row['song_id']
    if sid == seed_id:
        continue
    title = row['title']
    
    breakdown = multi.get_signal_breakdown(seed_id, sid)
    total = multi.calculate_similarity(seed_id, sid)
    
    results.append({
        'title': title,
        'total': total,
        'lyrics': breakdown.get('lyrics'),
        'audio': breakdown.get('audio'),
        'metadata': breakdown.get('metadata'),
    })

results.sort(key=lambda x: x['total'], reverse=True)

print(f'| # | 곡명 | 종합% | 가사 | 오디오 | 메타 |')
print(f'|--:|:---|---:|---:|---:|---:|')
for i, r in enumerate(results, 1):
    t = r['title'][:45]
    total = f"{r['total']*100:.1f}" if r['total'] else '-'
    ly = f"{r['lyrics']:.3f}" if r['lyrics'] is not None else '-'
    au = f"{r['audio']:.3f}" if r['audio'] is not None else '-'
    me = f"{r['metadata']:.2f}" if r['metadata'] is not None else '-'
    print(f'| {i} | {t} | {total} | {ly} | {au} | {me} |')
