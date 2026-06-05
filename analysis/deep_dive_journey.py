import pandas as pd
import numpy as np
import codecs, sys

sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

df = pd.read_csv(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# =============================================
# 분석 1: 개별 곡의 "여정(Journey)" 추적
# 곡을 충분히 많이 들은 상위 곡들에 대해
# 1번째, 2번째, 3번째... 들을 때마다 스킵했는지 안했는지를 추적
# =============================================

# 최소 20회 이상 재생된 곡만 대상
song_counts = df['song_id'].value_counts()
songs_over_20 = song_counts[song_counts >= 20].index

print(f"=== 20회 이상 재생된 곡 수: {len(songs_over_20)}곡 ===\n")

# 각 곡의 n번째 재생 시 스킵 여부를 추적
journey_data = []

for song in songs_over_20:
    song_df = df[df['song_id'] == song].sort_values('timestamp').reset_index(drop=True)
    for i, row in song_df.iterrows():
        journey_data.append({
            'song_id': song,
            'play_number': i + 1,  # 1번째, 2번째, ...
            'is_skipped': row['is_skipped'],
            'satisfaction_score': row['satisfaction_score'],
        })

journey_df = pd.DataFrame(journey_data)

# play_number별 평균 스킵률 (1번째 vs 2번째 vs ... vs 20번째)
print("=== 개별 곡 단위: N번째 재생 시 스킵률 ===")
print("(20회 이상 들은 곡들만 대상, 각 곡의 1번째/2번째/.../20번째 플레이의 스킵 여부)\n")

for n in range(1, 21):
    nth_plays = journey_df[journey_df['play_number'] == n]
    skip_rate = nth_plays['is_skipped'].mean()
    satisfaction = nth_plays['satisfaction_score'].mean()
    count = len(nth_plays)
    bar = '█' * int((1 - skip_rate) * 30)
    print(f"  {n:2d}번째 플레이: 스킵률 {skip_rate:.1%} | 만족도 {satisfaction:.2f} | {bar} ({count}곡)")

# =============================================
# 분석 2: "처음에 스킵했다가 나중에 안 스킵하게 된 곡"이 실제로 있는가?
# =============================================
print("\n=== 패턴 분류: 곡별 여정(Journey) 타입 ===\n")

pattern_counts = {'loved_from_start': 0, 'grew_on_me': 0, 'always_meh': 0, 'loved_then_tired': 0}
example_songs = {'loved_from_start': [], 'grew_on_me': [], 'always_meh': [], 'loved_then_tired': []}

for song in songs_over_20:
    song_df = df[df['song_id'] == song].sort_values('timestamp').reset_index(drop=True)
    total_plays = len(song_df)
    
    # 전반부 (처음 절반)와 후반부 (나중 절반) 스킵률 비교
    half = total_plays // 2
    first_half_skip = song_df.iloc[:half]['is_skipped'].mean()
    second_half_skip = song_df.iloc[half:]['is_skipped'].mean()
    
    # 처음 3회 스킵률
    first_3_skip = song_df.iloc[:min(3, total_plays)]['is_skipped'].mean()
    # 전체 스킵률
    overall_skip = song_df['is_skipped'].mean()
    
    song_name_short = song[:40]
    
    if first_3_skip <= 0.33 and overall_skip <= 0.3:
        pattern_counts['loved_from_start'] += 1
        if len(example_songs['loved_from_start']) < 3:
            example_songs['loved_from_start'].append(f"{song_name_short} ({total_plays}회)")
    elif first_half_skip > second_half_skip + 0.15:
        pattern_counts['grew_on_me'] += 1
        if len(example_songs['grew_on_me']) < 3:
            example_songs['grew_on_me'].append(f"{song_name_short} (전반 스킵 {first_half_skip:.0%} → 후반 {second_half_skip:.0%}, {total_plays}회)")
    elif second_half_skip > first_half_skip + 0.15:
        pattern_counts['loved_then_tired'] += 1
        if len(example_songs['loved_then_tired']) < 3:
            example_songs['loved_then_tired'].append(f"{song_name_short} (전반 스킵 {first_half_skip:.0%} → 후반 {second_half_skip:.0%}, {total_plays}회)")
    else:
        pattern_counts['always_meh'] += 1
        if len(example_songs['always_meh']) < 3:
            example_songs['always_meh'].append(f"{song_name_short} (스킵률 {overall_skip:.0%}, {total_plays}회)")

total_songs = sum(pattern_counts.values())
for pattern, count in pattern_counts.items():
    label = {
        'loved_from_start': '🟢 처음부터 좋았던 곡 (첫 3회 스킵 거의 안함, 꾸준히 좋음)',
        'grew_on_me': '🟡 듣다 보니 좋아진 곡 (전반부 스킵 多 → 후반부 스킵 少)',
        'always_meh': '⚪ 그냥저냥인 곡 (전반/후반 스킵률 비슷)',
        'loved_then_tired': '🔴 좋았다가 질린 곡 (전반부 스킵 少 → 후반부 스킵 多)',
    }[pattern]
    print(f"  {label}")
    print(f"    → {count}곡 ({count/total_songs*100:.1f}%)")
    for ex in example_songs[pattern]:
        print(f"      예: {ex}")
    print()
