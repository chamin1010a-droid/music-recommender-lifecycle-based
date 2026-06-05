"""
사용자의 체감 검증: 잔나비 곡별 생애주기 패턴 분석
- 입문곡(뜨거운 여름밤, 주저하는 연인들) → Zone 3 (좋다가 질림) 패턴인가?
- 신곡(애프터스쿨 액티비티, 마더 등) → Zone 2 (듣다보니 좋아짐) or Rising 패턴인가?
- 아티스트 전체 흐름과 개별 곡 흐름의 상호작용 확인
"""
import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

history_csv = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
df = pd.read_csv(history_csv, encoding='utf-8-sig')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 아티스트 정규화 (- Topic 제거)
def normalize_artist(name):
    if isinstance(name, str):
        name = name.replace(' - Topic', '').strip()
    return name

df['artist_norm'] = df['artist'].apply(normalize_artist)

# =============================================
# 1. 잔나비 전곡 생애주기 패턴 분석
# =============================================
jannabi = df[df['artist_norm'].str.contains('JANNABI', case=False, na=False)]

print("=" * 80)
print("🎸 잔나비(JANNABI) 전곡 생애주기 분석")
print("=" * 80)

jannabi_songs = {}
for song_id, group in jannabi.groupby('song_id'):
    group = group.sort_values('timestamp')
    title = group['title'].iloc[0]
    total = len(group)
    if total < 3:
        continue

    overall_skip = group['is_skipped'].mean()
    half = len(group) // 2
    first_half_skip = group.iloc[:half]['is_skipped'].mean()
    second_half_skip = group.iloc[half:]['is_skipped'].mean()
    first_3_skip = group.iloc[:min(3, len(group))]['is_skipped'].mean()
    
    # 최근 30일 vs 그 이전 스킵률
    ref = df['timestamp'].max()
    recent_30d = group[group['timestamp'] >= (ref - pd.Timedelta(days=30))]
    older = group[group['timestamp'] < (ref - pd.Timedelta(days=30))]
    recent_skip = recent_30d['is_skipped'].mean() if len(recent_30d) > 0 else None
    older_skip = older['is_skipped'].mean() if len(older) > 0 else None
    recent_plays = len(recent_30d)
    
    # 첫 재생일, 마지막 재생일
    first_play = group['timestamp'].min()
    last_play = group['timestamp'].max()
    days_since_last = (ref - last_play).days
    
    # Zone 분류
    if first_3_skip <= 0.33 and overall_skip <= 0.3:
        zone = 'Zone1(처음부터좋아함)'
    elif first_half_skip > second_half_skip + 0.15:
        zone = 'Zone2(듣다보니좋아짐)'
    elif second_half_skip > first_half_skip + 0.15:
        zone = 'Zone3(좋다가질림)'
    else:
        zone = 'Zone4(그냥저냥)'
    
    jannabi_songs[song_id] = {
        'title': title,
        'total': total,
        'overall_skip': overall_skip,
        'first_half_skip': first_half_skip,
        'second_half_skip': second_half_skip,
        'recent_30d_plays': recent_plays,
        'recent_30d_skip': recent_skip,
        'older_skip': older_skip,
        'days_since_last': days_since_last,
        'first_play': first_play,
        'last_play': last_play,
        'zone': zone
    }

# 총 재생수 기준 정렬
sorted_songs = sorted(jannabi_songs.values(), key=lambda x: x['total'], reverse=True)

print(f"\n총 {len(sorted_songs)}곡 (재생 3회 이상)")
print("-" * 80)
print(f"{'곡명':<35} {'재생':>4} {'전체':>5} {'전반':>5} {'후반':>5} {'최근30일':>8} {'Zone분류':<20}")
print("-" * 80)

for s in sorted_songs:
    recent_str = f"{s['recent_30d_plays']}회"
    if s['recent_30d_skip'] is not None:
        recent_str += f"({s['recent_30d_skip']*100:.0f}%skip)"
    else:
        recent_str += "(없음)"
    
    print(f"{s['title'][:34]:<35} {s['total']:>4} {s['overall_skip']*100:>4.0f}% {s['first_half_skip']*100:>4.0f}% {s['second_half_skip']*100:>4.0f}% {recent_str:>14} {s['zone']:<20}")

# =============================================
# 2. 사용자가 말한 입문곡 vs 신곡 비교
# =============================================
print("\n\n" + "=" * 80)
print("📌 사용자 체감 검증: 입문곡 vs 최근 꽂힌 곡")
print("=" * 80)

entry_keywords = ['뜨거운 여름밤', '주저하는', 'for lovers who hesitate', 'Summer']
new_keywords = ['LEGEND', 'Summer ll', 'Sweet memories', '밤의 공원', '그 밤 그 밤']

print("\n[입문곡 (초기에 많이 듣다 질린 곡들)]")
for s in sorted_songs:
    if any(kw in s['title'] for kw in entry_keywords):
        trend = "⬆️" if (s['recent_30d_skip'] is not None and s['older_skip'] is not None and s['recent_30d_skip'] < s['older_skip']) else "⬇️" if (s['recent_30d_skip'] is not None and s['older_skip'] is not None and s['recent_30d_skip'] > s['older_skip']) else "➡️"
        print(f"  {trend} {s['title'][:40]}")
        print(f"     총 {s['total']}회 | 전반 스킵 {s['first_half_skip']*100:.0f}% → 후반 스킵 {s['second_half_skip']*100:.0f}% | {s['zone']}")
        print(f"     최근30일: {s['recent_30d_plays']}회 재생 | 마지막 재생: {s['days_since_last']}일 전")

print("\n[최근 꽂힌 곡들 (신곡/신규 발견)]")
for s in sorted_songs:
    if any(kw in s['title'] for kw in new_keywords):
        trend = "⬆️" if (s['recent_30d_skip'] is not None and s['older_skip'] is not None and s['recent_30d_skip'] < s['older_skip']) else "⬇️" if (s['recent_30d_skip'] is not None and s['older_skip'] is not None and s['recent_30d_skip'] > s['older_skip']) else "➡️"
        print(f"  {trend} {s['title'][:40]}")
        print(f"     총 {s['total']}회 | 전반 스킵 {s['first_half_skip']*100:.0f}% → 후반 스킵 {s['second_half_skip']*100:.0f}% | {s['zone']}")
        print(f"     최근30일: {s['recent_30d_plays']}회 재생 | 마지막 재생: {s['days_since_last']}일 전")

# =============================================
# 3. 아티스트 전체 흐름: 최근 30일 잔나비 총 재생 추이
# =============================================
print("\n\n" + "=" * 80)
print("📊 잔나비 아티스트 전체 흐름 (주간 재생수 추이)")
print("=" * 80)

jannabi_sorted = jannabi.sort_values('timestamp')
jannabi_sorted['week'] = jannabi_sorted['timestamp'].dt.isocalendar().week.astype(int)
jannabi_sorted['year'] = jannabi_sorted['timestamp'].dt.year

# 최근 12주 추이
ref = jannabi_sorted['timestamp'].max()
recent_12w = jannabi_sorted[jannabi_sorted['timestamp'] >= (ref - pd.Timedelta(weeks=12))]
weekly = recent_12w.groupby([recent_12w['timestamp'].dt.isocalendar().year, recent_12w['timestamp'].dt.isocalendar().week]).agg(
    plays=('song_id', 'count'),
    skips=('is_skipped', 'sum'),
    skip_rate=('is_skipped', 'mean')
).reset_index()

print(f"{'주차':<12} {'재생수':>6} {'스킵수':>6} {'스킵률':>7}")
print("-" * 35)
for _, row in weekly.iterrows():
    bar = "█" * int(row['plays'] / 3)
    print(f"W{row['week']:>2}        {int(row['plays']):>6} {int(row['skips']):>6} {row['skip_rate']*100:>5.1f}%  {bar}")

# =============================================
# 4. 아티스트별 "남은 수명" 지표 (전체 곡 중 아직 살아있는 비중)
# =============================================
print("\n\n" + "=" * 80)
print("🏥 주요 아티스트별 '곡 생존율' (전체 곡 중 Frozen이 아닌 곡의 비율)")
print("=" * 80)

# 전곡 비교 CSV 로드
comparison = pd.read_csv(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\full_classification_comparison.csv', encoding='utf-8-sig')

top_artists = comparison.groupby('아티스트').agg(
    total_songs=('곡명', 'count'),
    alive=('곡명', lambda x: (comparison.loc[x.index, '[현재] 온도등급'].isin(['Rising', 'Steady', 'Warm'])).sum())
).reset_index()
top_artists['survival_rate'] = (top_artists['alive'] / top_artists['total_songs'] * 100).round(1)
top_artists = top_artists[top_artists['total_songs'] >= 5].sort_values('total_songs', ascending=False)

print(f"\n{'아티스트':<25} {'전체곡':>6} {'활성곡':>6} {'생존율':>7}")
print("-" * 50)
for _, row in top_artists.head(20).iterrows():
    bar = "█" * int(row['survival_rate'] / 5)
    status = "🟢" if row['survival_rate'] > 50 else "🟡" if row['survival_rate'] > 20 else "🔴"
    artist_short = row['아티스트'][:24]
    print(f"{status} {artist_short:<23} {int(row['total_songs']):>6} {int(row['alive']):>6} {row['survival_rate']:>6.1f}% {bar}")
