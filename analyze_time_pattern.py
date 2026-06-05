import sys, os, io
import pandas as pd
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

csv_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
df = pd.read_csv(csv_p)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])
df['hour'] = df['timestamp'].dt.hour

def get_period(h):
    if 0 <= h < 6: return '🌙 심야(0-6)'
    elif 6 <= h < 12: return '☀️ 오전(6-12)'
    elif 12 <= h < 18: return '🌤 오후(12-18)'
    else: return '🌆 저녁(18-24)'

df['period'] = df['hour'].apply(get_period)

# 1) 시간대별 전체 재생 분포
print("=" * 70)
print("📊 시간대별 청취 분포")
print("=" * 70)
period_counts = df['period'].value_counts().sort_index()
total = len(df)
for period, count in period_counts.items():
    bar = '█' * int(count / total * 50)
    print(f"  {period}  {count:>5}회 ({count/total*100:>5.1f}%)  {bar}")

# 2) 시간대별 장르 분포 (장르 데이터가 있다면)
# 대신 아티스트별 시간대 분포를 봐보자
print(f"\n{'=' * 70}")
print("📊 주요 아티스트별 시간대 편향 분석")
print("    (특정 시간대에 몰려있는 아티스트가 있는가?)")
print("=" * 70)

top_artists = df['artist'].value_counts().head(10).index.tolist()

for artist in top_artists:
    adf = df[df['artist'] == artist]
    period_dist = adf['period'].value_counts(normalize=True).sort_index()
    
    name = artist.replace(' - Topic', '')[:20]
    dominant = period_dist.idxmax()
    dominant_pct = period_dist.max() * 100
    
    # 전체 평균과 비교
    overall_dist = df['period'].value_counts(normalize=True).sort_index()
    max_deviation = 0
    most_different_period = ''
    for p in period_dist.index:
        dev = abs(period_dist.get(p, 0) - overall_dist.get(p, 0))
        if dev > max_deviation:
            max_deviation = dev
            most_different_period = p
    
    print(f"\n  🎤 {name}")
    for p in sorted(period_dist.index):
        pct = period_dist.get(p, 0) * 100
        overall_pct = overall_dist.get(p, 0) * 100
        diff = pct - overall_pct
        indicator = '⬆' if diff > 3 else '⬇' if diff < -3 else ' '
        print(f"     {p}: {pct:5.1f}% (전체평균 {overall_pct:.1f}%, 편차 {diff:+.1f}%) {indicator}")

# 3) 시간대별 스킵률 변화 (시간대가 기분에 영향을 미치는지)
if 'is_skipped' in df.columns:
    print(f"\n{'=' * 70}")
    print("📊 시간대별 스킵률 (시간대에 따라 청취 품질이 다른가?)")
    print("=" * 70)
    skip_by_period = df.groupby('period')['is_skipped'].agg(['mean', 'count']).sort_index()
    for period, row in skip_by_period.iterrows():
        print(f"  {period}  스킵률: {row['mean']*100:5.1f}%  (n={row['count']})")

# 4) 특정 곡이 특정 시간대에만 들리는 경향이 있는가?
print(f"\n{'=' * 70}")
print("📊 시간대 편향이 가장 큰 곡 TOP 10")
print("    (특정 시간에만 집중적으로 듣는 곡)")
print("=" * 70)

song_groups = df.groupby('song_id')
biased_songs = []
for song_id, group in song_groups:
    if len(group) < 10:  # 최소 10회 이상 들은 곡만
        continue
    period_dist = group['period'].value_counts(normalize=True)
    # 가장 많이 듣는 시간대 비율이 70% 이상이면 편향
    max_pct = period_dist.max()
    dominant_p = period_dist.idxmax()
    title = group['title'].iloc[0]
    artist = group['artist'].iloc[0].replace(' - Topic', '')
    biased_songs.append({
        'title': title, 'artist': artist,
        'plays': len(group), 'dominant': dominant_p,
        'pct': max_pct, 'song_id': song_id
    })

biased_songs.sort(key=lambda x: x['pct'], reverse=True)
for s in biased_songs[:10]:
    print(f"  {s['artist'][:15]:<16} {s['title'][:30]:<32} {s['plays']:>3}회 → {s['dominant']} {s['pct']*100:.0f}%")

# 5) 카이제곱 검정: 아티스트별 시간대 분포가 전체 분포와 유의하게 다른가?
from scipy.stats import chi2_contingency
print(f"\n{'=' * 70}")
print("📊 카이제곱 검정: 아티스트별 시간대 분포 독립성 검정")
print("=" * 70)
for artist in top_artists[:5]:
    adf = df[df['artist'] == artist]
    ct = pd.crosstab(df['artist'] == artist, df['period'])
    chi2, p, dof, expected = chi2_contingency(ct)
    name = artist.replace(' - Topic', '')[:20]
    sig = "✅ 유의미" if p < 0.05 else "❌ 유의하지 않음"
    print(f"  {name:<22} χ²={chi2:8.2f}  p={p:.4f}  {sig}")

print("\nDONE")
