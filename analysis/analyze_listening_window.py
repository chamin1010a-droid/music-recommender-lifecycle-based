"""
30일 vs 14일 기준 청취 패턴 분석
- 전체 청취 시간, 곡 수, 가수 수
- 특정 가수별 상세 비교
"""
import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

history_csv = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
df = pd.read_csv(history_csv, encoding='utf-8-sig')
df['timestamp'] = pd.to_datetime(df['timestamp'])

def normalize_artist(name):
    if isinstance(name, str):
        name = name.replace(' - Topic', '').strip()
    return name

df['artist_norm'] = df['artist'].apply(normalize_artist)

ref = df['timestamp'].max()
print(f"기준일: {ref.strftime('%Y-%m-%d')}")

# =============================================
# 1. 전체 청취 통계 (30일 vs 14일)
# =============================================
for days, label in [(30, '30일'), (14, '14일')]:
    period = df[df['timestamp'] >= (ref - pd.Timedelta(days=days))]
    total_plays = len(period)
    total_skips = period['is_skipped'].sum()
    unique_songs = period['song_id'].nunique()
    unique_artists = period['artist_norm'].nunique()
    
    # 곡당 평균 3.5분으로 추정 (스킵은 0.5분)
    full_listens = total_plays - total_skips
    est_hours = (full_listens * 3.5 + total_skips * 0.5) / 60
    
    print(f"\n{'='*60}")
    print(f"📊 최근 {label} 전체 청취 통계")
    print(f"{'='*60}")
    print(f"  총 재생 횟수:  {total_plays}회")
    print(f"  총 스킵 횟수:  {int(total_skips)}회 (스킵률 {total_skips/total_plays*100:.1f}%)")
    print(f"  고유 곡 수:    {unique_songs}곡")
    print(f"  고유 가수 수:  {unique_artists}명")
    print(f"  추정 청취시간: 약 {est_hours:.1f}시간")
    print(f"  하루 평균:     약 {total_plays/days:.0f}회 재생 / 약 {est_hours/days:.1f}시간")

# =============================================
# 2. 특정 가수별 30일 vs 14일 비교
# =============================================
target_artists = {
    '잔나비': ['JANNABI'],
    '검정치마': ['The Black Skirts', 'The Black Skirt'],
    '카더가든': ['Car, the Garden'],
    'Xdinary Heroes': ['Xdinary Heroes'],
    '찰리푸스': ['Charlie Puth'],
    '오아시스': ['Oasis'],
    'SG워너비': ['SG Wannabe'],
    '엠씨더맥스': ['M.C the MAX', 'MC the MAX'],
    '한로로': ['HANRORO'],
}

print(f"\n\n{'='*80}")
print(f"🎤 가수별 30일 vs 14일 청취 비교")
print(f"{'='*80}")
print(f"{'가수':<15} {'| 30일 재생':>10} {'스킵률':>7} {'고유곡':>6} {'| 14일 재생':>10} {'스킵률':>7} {'고유곡':>6} {'| 변화':>8}")
print("-" * 80)

for display_name, keywords in target_artists.items():
    artist_df = df[df['artist_norm'].apply(lambda x: any(kw.lower() in str(x).lower() for kw in keywords))]
    
    for days, label in [(30, '30d'), (14, '14d')]:
        period = artist_df[artist_df['timestamp'] >= (ref - pd.Timedelta(days=days))]
        plays = len(period)
        skips = period['is_skipped'].sum() if plays > 0 else 0
        skip_rate = skips / plays * 100 if plays > 0 else 0
        unique = period['song_id'].nunique()
        
        if label == '30d':
            p30, s30, u30 = plays, skip_rate, unique
        else:
            p14, s14, u14 = plays, skip_rate, unique
    
    # 변화 판단
    if p30 > 0:
        recent_ratio = p14 / (p30 / 30 * 14) if p30 > 0 else 0
        if recent_ratio > 1.3:
            trend = "📈 증가"
        elif recent_ratio < 0.7:
            trend = "📉 감소"
        else:
            trend = "➡️ 유지"
    else:
        trend = "⛔ 없음"
    
    print(f"{display_name:<15} | {p30:>6}회 {s30:>5.1f}% {u30:>4}곡 | {p14:>6}회 {s14:>5.1f}% {u14:>4}곡 | {trend}")

# =============================================
# 3. 가수별 상세: 최근 30일 내 곡별 재생 패턴 (Top 5 곡)
# =============================================
print(f"\n\n{'='*80}")
print(f"🔍 가수별 최근 30일 내 Top 5 곡 (재생수 기준)")
print(f"{'='*80}")

for display_name, keywords in target_artists.items():
    artist_df = df[df['artist_norm'].apply(lambda x: any(kw.lower() in str(x).lower() for kw in keywords))]
    recent = artist_df[artist_df['timestamp'] >= (ref - pd.Timedelta(days=30))]
    
    if len(recent) == 0:
        print(f"\n--- {display_name}: 최근 30일 재생 없음 ---")
        continue
    
    song_stats = recent.groupby('song_id').agg(
        title=('title', 'first'),
        plays=('song_id', 'count'),
        skips=('is_skipped', 'sum')
    ).reset_index()
    song_stats['skip_rate'] = (song_stats['skips'] / song_stats['plays'] * 100).round(1)
    song_stats = song_stats.sort_values('plays', ascending=False)
    
    print(f"\n--- {display_name} (최근 30일: 총 {len(recent)}회 재생) ---")
    for _, row in song_stats.head(5).iterrows():
        print(f"  {row['title'][:35]:<37} {int(row['plays']):>3}회 (스킵 {row['skip_rate']:.0f}%)")
