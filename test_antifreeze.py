import sys, os
sys.path.append(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\core')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from lyrics_engine import LyricsEngine
from audio_features_engine import AudioFeaturesEngine
from multi_signal_engine import MultiSignalSimilarityEngine
from song_title_normalizer import SongTitleNormalizer

csv_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
meta_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\data\caches\ytm_metadata_cache.csv'
df = pd.read_csv(csv_p, encoding='utf-8-sig')

# 제목 노말라이저
norm = SongTitleNormalizer()

# 검정치마 곡
bs = df[df['artist'].str.contains('Black Skirts', case=False, na=False)][['song_id','title','artist']].drop_duplicates('song_id')

# 엔진 로드
lyrics_engine = LyricsEngine(genius_token='x')
lyrics_engine._build_matrix()
audio_engine = AudioFeaturesEngine()
audio_engine._build_matrix()
multi = MultiSignalSimilarityEngine(
    tag_engine=None, lyrics_engine=lyrics_engine,
    audio_engine=audio_engine, metadata_path=meta_p
)

seed_id = 'Antifreeze - The Black Skirts - Topic'
print(f"시드: 부동액 (Antifreeze) — 검정치마\n")

# 가사 + 종합 동시 계산
results = []
for _, row in bs.iterrows():
    sid = row['song_id']
    if sid == seed_id:
        continue
    
    lyrics_sim = lyrics_engine.calculate_similarity(seed_id, sid)
    total_sim = multi.calculate_similarity(seed_id, sid)
    breakdown = multi.get_signal_breakdown(seed_id, sid)
    
    kr = norm.get_display_title(sid)
    en = row['title']
    display = f"{kr}" if kr != en else en
    
    results.append({
        'title': display,
        'title_en': en,
        'lyrics': lyrics_sim,
        'audio': breakdown.get('audio'),
        'meta': breakdown.get('metadata'),
        'total': total_sim,
    })

# === 가사 유사도 순위 ===
by_lyrics = sorted(results, key=lambda x: x['lyrics'] if x['lyrics'] is not None else -1, reverse=True)
print("## 📝 가사 유사도 순위 (Antifreeze 기준)")
print(f"| # | 곡명 | 가사 |")
print(f"|--:|:---|---:|")
for i, r in enumerate(by_lyrics[:20], 1):
    t = r['title'][:40]
    ly = f"{r['lyrics']:.3f}" if r['lyrics'] is not None else '-'
    print(f"| {i} | {t} | {ly} |")

print(f"\n  ⋮ (하위 5곡)")
for i, r in enumerate(by_lyrics[-5:], len(by_lyrics)-4):
    t = r['title'][:40]
    ly = f"{r['lyrics']:.3f}" if r['lyrics'] is not None else '-'
    print(f"| {i} | {t} | {ly} |")

# === 종합 유사도 순위 ===
by_total = sorted(results, key=lambda x: x['total'], reverse=True)
print(f"\n\n## 🎯 종합 유사도 순위 (가사30%+오디오30%+메타15%+태그25%)")
print(f"| # | 곡명 | 종합% | 가사 | 오디오 | 메타 |")
print(f"|--:|:---|---:|---:|---:|---:|")
for i, r in enumerate(by_total[:20], 1):
    t = r['title'][:40]
    total = f"{r['total']*100:.1f}"
    ly = f"{r['lyrics']:.3f}" if r['lyrics'] is not None else '-'
    au = f"{r['audio']:.3f}" if r['audio'] is not None else '-'
    me = f"{r['meta']:.2f}" if r['meta'] is not None else '-'
    print(f"| {i} | {t} | {total} | {ly} | {au} | {me} |")

print(f"\n  ⋮ (하위 5곡)")
for i, r in enumerate(by_total[-5:], len(by_total)-4):
    t = r['title'][:40]
    total = f"{r['total']*100:.1f}"
    ly = f"{r['lyrics']:.3f}" if r['lyrics'] is not None else '-'
    au = f"{r['audio']:.3f}" if r['audio'] is not None else '-'
    me = f"{r['meta']:.2f}" if r['meta'] is not None else '-'
    print(f"| {i} | {t} | {total} | {ly} | {au} | {me} |")
