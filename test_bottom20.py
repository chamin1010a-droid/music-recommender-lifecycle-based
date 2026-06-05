import sys,os
sys.path.append(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\core')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from lyrics_engine import LyricsEngine
from audio_features_engine import AudioFeaturesEngine
from multi_signal_engine import MultiSignalSimilarityEngine
from song_title_normalizer import SongTitleNormalizer

csv_p=r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
meta_p=r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\data\caches\ytm_metadata_cache.csv'
df=pd.read_csv(csv_p,encoding='utf-8-sig')
norm=SongTitleNormalizer()
bs=df[df['artist'].str.contains('Black Skirts',case=False,na=False)][['song_id','title','artist']].drop_duplicates('song_id')
le=LyricsEngine(genius_token='x'); le._build_matrix()
ae=AudioFeaturesEngine(); ae._build_matrix()
m=MultiSignalSimilarityEngine(tag_engine=None,lyrics_engine=le,audio_engine=ae,metadata_path=meta_p)
seed='Antifreeze - The Black Skirts - Topic'
results=[]
for _,r in bs.iterrows():
    sid=r['song_id']
    if sid==seed: continue
    ls=le.calculate_similarity(seed,sid)
    ts=m.calculate_similarity(seed,sid)
    bd=m.get_signal_breakdown(seed,sid)
    kr=norm.get_display_title(sid); en=r['title']
    results.append({'t':kr if kr!=en else en,'lyrics':ls,'audio':bd.get('audio'),'meta':bd.get('metadata'),'total':ts})

by_lyrics=sorted(results,key=lambda x:x['lyrics'] if x['lyrics'] is not None else -1,reverse=True)
by_total=sorted(results,key=lambda x:x['total'],reverse=True)

print('## 가사 하위 20곡')
print('| # | 곡명 | 가사 |')
print('|--:|:---|---:|')
for i,r in enumerate(by_lyrics[-20:],len(by_lyrics)-19):
    ly=f"{r['lyrics']:.3f}" if r['lyrics'] is not None else '-'
    print(f"| {i} | {r['t'][:40]} | {ly} |")

print()
print('## 종합 하위 20곡')
print('| # | 곡명 | 종합% | 가사 | 오디오 | 메타 |')
print('|--:|:---|---:|---:|---:|---:|')
for i,r in enumerate(by_total[-20:],len(by_total)-19):
    ly=f"{r['lyrics']:.3f}" if r['lyrics'] is not None else '-'
    au=f"{r['audio']:.3f}" if r['audio'] is not None else '-'
    me=f"{r['meta']:.2f}" if r['meta'] is not None else '-'
    print(f"| {i} | {r['t'][:40]} | {r['total']*100:.1f} | {ly} | {au} | {me} |")
