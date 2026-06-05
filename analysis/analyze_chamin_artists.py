import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
csv_path = os.path.join(BASE_DIR, 'Takeout', 'YouTube 및 YouTube Music', '시청 기록', 'ytm_history_features.csv')

df = pd.read_csv(csv_path, encoding='utf-8-sig')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 1. user 아티스트 쇠퇴 패턴 및 유지 패턴 분석
df['week_num'] = (df['timestamp'] - df['timestamp'].min()).dt.days // 7

artist_stats = []

for artist, group in df.groupby('artist'):
    # 기본 필터링: 플레이 횟수 및 기간
    if len(group) < 15:
        continue
        
    weekly = group.groupby('week_num').size()
    if len(weekly) < 6:
        continue
        
    mid = len(weekly) // 2
    first_avg = weekly.iloc[:mid].mean()
    second_avg = weekly.iloc[mid:].mean()
    
    if first_avg < 1:
        continue
        
    decline_ratio = second_avg / first_avg
    
    artist_name = str(artist).replace(' - Topic', '')
    
    is_declining = False
    if decline_ratio <= 0.4 and first_avg >= 2:
        is_declining = True
        
    artist_stats.append({
        'artist': artist_name,
        'total_plays': len(group),
        'first_half_weekly': first_avg,
        'second_half_weekly': second_avg,
        'recent_30d': len(group[group['timestamp'] >= (df['timestamp'].max() - pd.Timedelta(days=30))]),
        'ratio': decline_ratio,
        'is_declining': is_declining
    })

# DataFrame 변환
stats_df = pd.DataFrame(artist_stats)

print("=" * 70)
print("🧑 user (본인) 아티스트 단위 쇠퇴 패턴 통계")
print("=" * 70)

# --- 1. 쇠퇴 패턴 (Decline) 보이는 아티스트 ---
declining = stats_df[stats_df['is_declining']].sort_values('total_plays', ascending=False)
print(f"\n📉 1. 전반기 대비 60% 이상 재생량이 증발한 아티스트 (총 {len(declining)}명)")
print(f"  {'아티스트':<22} {'총재생':>5} {'전반주간':>6} {'후반주간':>6} {'잔존율':>5} {'최근30일':>6}")
print("-" * 65)
for _, row in declining.iterrows():
    print(f"  {row['artist'][:20]:<22} {row['total_plays']:>5}회 {row['first_half_weekly']:>6.1f}회 {row['second_half_weekly']:>6.1f}회 {row['ratio']*100:>4.0f}% {row['recent_30d']:>4}회")

# --- 2. 쇠퇴하지 않은 아티스트 TOP 10 ---
stable = stats_df[~stats_df['is_declining']].sort_values('total_plays', ascending=False)
top10_stable = stable.head(10)

print(f"\n\n🔥 2. 쇠퇴 패턴을 보이지 않은 TOP 10 많이 듣는 아티스트")
print("   (수명이 유지되거나 심지어 '성장'하고 있는 핵심 아티스트들)")
print(f"  {'아티스트':<22} {'총재생':>5} {'전반주간':>6} {'후반주간':>6} {'잔존율':>5} {'최근30일':>6}")
print("-" * 65)
for _, row in top10_stable.iterrows():
    trend = "📈" if row['ratio'] > 1.2 else "➡️"
    print(f"{trend} {row['artist'][:20]:<22} {row['total_plays']:>5}회 {row['first_half_weekly']:>6.1f}회 {row['second_half_weekly']:>6.1f}회 {row['ratio']*100:>4.0f}% {row['recent_30d']:>4}회")

print("\n\n💡 결론 해석: 쇠퇴하는 가수들은 후반기/최근30일 재생이 완전히 메말라가는 반면, \n"
      "유지/성장하는 가수들(잔나비, 검정치마, 데이식스)은 꾸준히 방어하거나 오히려 재생량이 대폭 증가합니다.\n"
      "이것이 '곡' 단위가 아닌 '아티스트' 단위의 수명 주기 감지가 결정적으로 중요한 이유입니다.")
