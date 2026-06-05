"""
[청취 순서 × 스킵 상관관계 분석]

질문: "내가 어떤 곡을 듣고 난 뒤에 다음 곡을 넘길 확률이 달라지는가?"
특히: "신곡(재생 3회 이하)을 처음 들을 때, 바로 전에 어떤 등급의 곡을 들었느냐가 수용/거부에 영향을 주는가?"

분석 방법:
1. 시간순으로 정렬된 재생 기록에서 (이전곡, 다음곡) 페어를 만든다.
2. 각 곡에 현재 Zone 기반 온도를 부여한다.
3. "다음곡을 스킵했는가" vs "이전곡의 온도는 무엇이었는가"의 상관관계를 측정한다.
4. 특히 다음곡이 "신곡(재생 3회 이하)"인 경우에 집중하여 패턴을 찾는다.
"""

import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from lifecycle_recommender import run_pipeline

sys.stdout.reconfigure(encoding='utf-8')

csv_path = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
meta_path = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\ytm_metadata_cache.csv'

# === Step 1: 전곡 온도 분류 ===
print("Step 1: 전곡 온도 분류 중...")
result = run_pipeline(
    csv_path=csv_path,
    user_name='user',
    playlist_size=15,
    preset='default',
    metadata_path=meta_path,
    user_birth_year=1998
)
temps = result['temp_tracker'].song_temps

# song_id → temperature 매핑
song_temp_map = {sid: info['temperature'] for sid, info in temps.items()}
song_plays_map = {sid: info['total_plays'] for sid, info in temps.items()}

# === Step 2: 시간순 재생 기록에서 (이전곡, 다음곡) 페어 생성 ===
print("Step 2: 시간순 페어 생성 중...")
df = pd.read_csv(csv_path, encoding='utf-8-sig')
df['date'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('date').reset_index(drop=True)

# 연속 재생 판정: 이전곡과의 시간 차이가 10분 이내면 같은 세션
df['time_diff'] = df['date'].diff().dt.total_seconds()
df['same_session'] = df['time_diff'] <= 600  # 10분 이내

pairs = []
for i in range(1, len(df)):
    if not df.loc[i, 'same_session']:
        continue  # 세션이 끊기면 스킵 (다른 시간대에 다시 켠 것)
    
    prev_song = df.loc[i-1, 'song_id']
    curr_song = df.loc[i, 'song_id']
    curr_skipped = df.loc[i, 'is_skipped']
    
    prev_temp = song_temp_map.get(prev_song, 'Unknown')
    curr_temp = song_temp_map.get(curr_song, 'Unknown')
    curr_total_plays = song_plays_map.get(curr_song, 0)
    
    # 현재 곡이 "신곡"인지 (총 재생 3회 이하)
    is_new_song = curr_total_plays <= 3
    
    pairs.append({
        'prev_temp': prev_temp,
        'curr_temp': curr_temp,
        'curr_skipped': int(curr_skipped),
        'is_new_song': is_new_song,
        'curr_total_plays': curr_total_plays,
        'prev_song_id': prev_song,
        'curr_song_id': curr_song,
    })

pairs_df = pd.DataFrame(pairs)
print(f"  총 {len(pairs_df)}개의 연속 재생 페어 생성 완료\n")

# === Step 3: 분석 1 — "이전곡 온도별 다음곡 스킵률" ===
print("=" * 70)
print("📊 분석 1: 이전곡 온도별 → 다음곡 스킵률")
print("  (이전에 어떤 등급의 곡을 들었을 때 다음 곡을 덜 넘기는가?)")
print("=" * 70)

for prev_temp in ['Rising', 'Steady', 'Warm', 'Cool', 'Frozen']:
    subset = pairs_df[pairs_df['prev_temp'] == prev_temp]
    if len(subset) < 10:
        continue
    skip_rate = subset['curr_skipped'].mean()
    n = len(subset)
    print(f"  이전곡 [{prev_temp:>8}] 후 → 다음곡 스킵률: {skip_rate*100:5.1f}% ({n}건)")

# === Step 4: 분석 2 — "신곡의 수용/거부 vs 이전곡 온도" (핵심!) ===
print(f"\n{'=' * 70}")
print("📊 분석 2: 신곡(재생≤3) 수용/거부 vs 직전 곡 온도 (핵심!)")
print("  (어떤 곡을 듣고 나서 신곡이 나왔을 때 수용 확률이 높은가?)")
print("=" * 70)

new_song_pairs = pairs_df[pairs_df['is_new_song'] == True]
print(f"  신곡 등장 페어 총 {len(new_song_pairs)}건\n")

for prev_temp in ['Rising', 'Steady', 'Warm', 'Cool', 'Frozen']:
    subset = new_song_pairs[new_song_pairs['prev_temp'] == prev_temp]
    if len(subset) < 5:
        continue
    skip_rate = subset['curr_skipped'].mean()
    accept_rate = 1 - skip_rate
    n = len(subset)
    bar = '█' * int(accept_rate * 20) + '░' * (20 - int(accept_rate * 20))
    print(f"  이전곡 [{prev_temp:>8}] 후 신곡 수용률: {accept_rate*100:5.1f}% |{bar}| ({n}건)")

# === Step 5: 분석 3 — "신곡 자체의 속성이 더 중요한가?" ===
print(f"\n{'=' * 70}")
print("📊 분석 3: 신곡 수용 vs 거부 — 곡 자체 속성 비교")
print("  (어떤 신곡이 수용되었는가? → 위치보다 곡 자체가 중요한지 검증)")
print("=" * 70)

accepted_new = new_song_pairs[new_song_pairs['curr_skipped'] == 0]
rejected_new = new_song_pairs[new_song_pairs['curr_skipped'] == 1]

print(f"\n  수용된 신곡: {len(accepted_new)}건")
print(f"  거부된 신곡: {len(rejected_new)}건")

# 수용된 신곡들의 아티스트 분포
if len(accepted_new) > 0:
    acc_artists = []
    for _, row in accepted_new.iterrows():
        info = temps.get(row['curr_song_id'], {})
        acc_artists.append(info.get('artist', 'Unknown'))
    
    from collections import Counter
    acc_counter = Counter(acc_artists)
    print(f"\n  🟢 수용된 신곡의 아티스트 TOP 10:")
    for artist, cnt in acc_counter.most_common(10):
        artist_clean = artist.replace(' - Topic', '')
        print(f"    {artist_clean:<20} — {cnt}건")

if len(rejected_new) > 0:
    rej_artists = []
    for _, row in rejected_new.iterrows():
        info = temps.get(row['curr_song_id'], {})
        rej_artists.append(info.get('artist', 'Unknown'))
    
    rej_counter = Counter(rej_artists)
    print(f"\n  🔴 거부된 신곡의 아티스트 TOP 10:")
    for artist, cnt in rej_counter.most_common(10):
        artist_clean = artist.replace(' - Topic', '')
        print(f"    {artist_clean:<20} — {cnt}건")

# === Step 6: 분석 4 — "수용된 신곡" vs "거부된 신곡"의 아티스트 생존율 비교 ===
print(f"\n{'=' * 70}")
print("📊 분석 4: 수용 vs 거부 신곡 — 아티스트 생존율 차이")
print("  (내가 좋아하는 가수의 신곡이면 수용 확률이 더 높은가?)")
print("=" * 70)

artist_survival = result['temp_tracker'].artist_survival

acc_survivals = []
for _, row in accepted_new.iterrows():
    info = temps.get(row['curr_song_id'], {})
    surv = artist_survival.get(info.get('artist', ''), 0)
    acc_survivals.append(surv)

rej_survivals = []
for _, row in rejected_new.iterrows():
    info = temps.get(row['curr_song_id'], {})
    surv = artist_survival.get(info.get('artist', ''), 0)
    rej_survivals.append(surv)

if acc_survivals and rej_survivals:
    print(f"  수용된 신곡의 아티스트 평균 생존율: {np.mean(acc_survivals)*100:.1f}%")
    print(f"  거부된 신곡의 아티스트 평균 생존율: {np.mean(rej_survivals)*100:.1f}%")
    diff = np.mean(acc_survivals) - np.mean(rej_survivals)
    if diff > 0.05:
        print(f"\n  📌 결론: 아티스트 생존율이 {diff*100:.1f}%p 더 높은 곡이 수용됨 → 곡 자체(아티스트 친밀도)가 위치보다 중요!")
    elif diff < -0.05:
        print(f"\n  📌 결론: 의외로 생존율이 낮은 아티스트의 곡도 수용되고 있음 → 위치 또는 다른 요인이 중요할 수 있음")
    else:
        print(f"\n  📌 결론: 생존율 차이 {diff*100:.1f}%p → 유의미한 차이 없음. 복합 요인.")

# === Step 7: 카이제곱 검정 — 이전곡 온도와 신곡 스킵의 통계적 독립성 ===
print(f"\n{'=' * 70}")
print("📊 분석 5: 통계 검정 — 이전곡 온도 ↔ 신곡 스킵 독립성")
print("=" * 70)

try:
    from scipy.stats import chi2_contingency
    
    # 이전곡 온도 × 신곡 스킵 여부 교차표
    valid_temps = ['Rising', 'Steady', 'Warm', 'Cool', 'Frozen']
    filtered = new_song_pairs[new_song_pairs['prev_temp'].isin(valid_temps)]
    
    if len(filtered) > 20:
        ct = pd.crosstab(filtered['prev_temp'], filtered['curr_skipped'])
        chi2, p, dof, expected = chi2_contingency(ct)
        
        print(f"  카이제곱 통계량: {chi2:.2f}")
        print(f"  p-value: {p:.4f}")
        print(f"  자유도: {dof}")
        
        if p < 0.05:
            print(f"\n  📌 p < 0.05 → 이전곡 온도가 신곡 스킵에 통계적으로 유의미한 영향이 있음!")
        else:
            print(f"\n  📌 p = {p:.4f} ≥ 0.05 → 이전곡 온도와 신곡 스킵은 통계적으로 독립.")
            print(f"     → '어떤 곡 뒤에 신곡이 나오냐'보다 '어떤 신곡이냐'가 더 중요하다는 의미!")
    else:
        print("  데이터 부족으로 검정 불가")
except ImportError:
    print("  scipy 없어서 카이제곱 검정 생략")

print(f"\n{'=' * 70}")
print("✅ 분석 완료!")
print("=" * 70)
