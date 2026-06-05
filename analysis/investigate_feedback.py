import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

csv_path = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
df = pd.read_csv(csv_path, encoding='utf-8-sig')
df['date'] = pd.to_datetime(df['timestamp'])

# ============ 오아시스 전체 데이터 확인 ============
print("=" * 70)
print("🔍 오아시스(Oasis) 데이터 전수 조사")
print("=" * 70)

# 오아시스 관련 아티스트명 모두 찾기
oasis_mask = df['artist'].str.contains('Oasis|oasis', case=False, na=False)
oasis_df = df[oasis_mask].copy()

print(f"\n총 재생 기록: {len(oasis_df)}건")
print(f"고유 곡(song_id) 수: {oasis_df['song_id'].nunique()}")
print(f"기간: {oasis_df['date'].min()} ~ {oasis_df['date'].max()}")

# 아티스트명 변형 확인
print(f"\n아티스트명 종류: {oasis_df['artist'].unique()}")

# 곡별 재생 횟수
print("\n--- 오아시스 곡별 재생 횟수 (전체) ---")
song_stats = oasis_df.groupby(['title', 'song_id']).agg(
    plays=('date', 'count'),
    first_play=('date', 'min'),
    last_play=('date', 'max'),
).sort_values('plays', ascending=False)

for _, row in song_stats.head(20).iterrows():
    title = _[0][:55]
    print(f"  {row['plays']:>3}회 | {str(row['last_play'])[:10]} | {title}")

# ============ 라이브 영상 vs 일반 곡 분리 ============
print("\n\n--- 라이브 영상 vs 일반 곡 ---")
live_mask = oasis_df['title'].str.contains('Live|live|Concert|concert|Unplugged|LIVE', na=False)
live_df = oasis_df[live_mask]
normal_df = oasis_df[~live_mask]

print(f"라이브 영상: {len(live_df)}건, 고유 {live_df['song_id'].nunique()}개")
print(f"일반 곡:     {len(normal_df)}건, 고유 {normal_df['song_id'].nunique()}개")

if len(live_df) > 0:
    print("\n라이브 영상 목록:")
    for _, row in live_df.groupby('title').size().sort_values(ascending=False).head(10).items():
        print(f"  {row:>3}회 | {_[:60]}")

# ============ time_gap 분석 (라이브 영상은 time_gap이 클 것) ============
print("\n\n--- time_gap_seconds 분석 (라이브 영상의 재생 시간이 길었는지 확인) ---")
if 'time_gap_seconds' in df.columns:
    oasis_df['time_gap'] = pd.to_numeric(oasis_df['time_gap_seconds'], errors='coerce')
    
    print(f"오아시스 전체 평균 time_gap: {oasis_df['time_gap'].mean():.0f}초")
    print(f"오아시스 전체 중간값 time_gap: {oasis_df['time_gap'].median():.0f}초")
    
    if len(live_df) > 0:
        live_gaps = oasis_df[oasis_df.index.isin(live_df.index)]['time_gap']
        normal_gaps = oasis_df[~oasis_df.index.isin(live_df.index)]['time_gap']
        print(f"\n라이브 평균 time_gap: {live_gaps.mean():.0f}초 ({live_gaps.mean()/60:.1f}분)")
        print(f"일반곡 평균 time_gap: {normal_gaps.mean():.0f}초 ({normal_gaps.mean()/60:.1f}분)")

# ============ 비교: 잔나비 & 검정치마의 총 재생 수 ============
print("\n\n--- 비교: 주요 아티스트 총 재생 횟수 ---")
for artist_keyword in ['JANNABI', 'Black Skirts', 'Oasis', 'Charlie Puth']:
    mask = df['artist'].str.contains(artist_keyword, case=False, na=False)
    adf = df[mask]
    unique_songs = adf['song_id'].nunique()
    total_plays = len(adf)
    print(f"  {artist_keyword:<15} 총 {total_plays:>5}회 재생 | 고유 곡 {unique_songs:>3}개 | 곡당 평균 {total_plays/max(unique_songs,1):.1f}회")

# ============ 찰리푸스 Marvin Gaye vs The Way I Am 비교 ============
print("\n\n--- 찰리푸스: Marvin Gaye vs The Way I Am ---")
for keyword in ['Marvin Gaye', 'The Way I Am']:
    mask = df['title'].str.contains(keyword, case=False, na=False)
    sdf = df[mask].sort_values('date')
    if len(sdf) > 0:
        print(f"\n  [{keyword}] 총 {len(sdf)}회")
        print(f"  최근 5회 재생일:")
        for _, row in sdf.tail(5).iterrows():
            print(f"    {str(row['date'])[:16]} | 스킵: {row['is_skipped']}")
