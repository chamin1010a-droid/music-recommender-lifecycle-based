import pandas as pd
import numpy as np
import os
import sys

# Set stdout to utf-8
sys.stdout.reconfigure(encoding='utf-8')

# User's data path - based on previous conversation context
user_csv = r"c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv"

if not os.path.exists(user_csv):
    # Try alternate path if not found
    user_csv = r"c:\Users\user\Desktop\데이터분석\음악 프로젝트\ytm_history_features.csv"

if not os.path.exists(user_csv):
     print("Error: User data file not found.")
     sys.exit(1)

df = pd.read_csv(user_csv, encoding='utf-8-sig')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 1. Interval calculation
df = df.sort_values(['song_id', 'timestamp'])
df['interval_h'] = (df.groupby('song_id')['timestamp'].shift(-1) - df['timestamp']).dt.total_seconds() / 3600

# 2. Activity calculation (weekly)
df['week'] = df['timestamp'].dt.to_period('W').apply(lambda r: r.start_time)
weekly_activity = df.groupby('week')['song_id'].count().rename('activity')
df = df.merge(weekly_activity, on='week', how='left')

# 3. Filter songs with 15+ plays
song_counts = df['song_id'].value_counts()
target_songs = song_counts[song_counts >= 15].index

results = []
for song in target_songs:
    s_df = df[df['song_id'] == song].sort_values('timestamp').reset_index(drop=True)
    n = len(s_df)
    
    # Early (First 5) vs Late (Last 5)
    e = s_df.head(5)
    l = s_df.tail(5)
    
    sat_e, sat_l = e['satisfaction_score'].mean(), l['satisfaction_score'].mean()
    int_e, int_l = e['interval_h'].median(), l['interval_h'].median()
    act_e, act_l = e['activity'].mean(), l['activity'].mean()
    
    results.append({
        'song': song,
        'plays': n,
        'sat_e': sat_e, 'sat_l': sat_l,
        'int_e': int_e, 'int_l': int_l,
        'act_e': act_e, 'act_l': act_l
    })

res_df = pd.DataFrame(results)

# Categorization
# 1. 질려하는 곡 (Burnout): Sat Drop + Int Increase + Act maintained
burnout = res_df[
    (res_df['sat_l'] < res_df['sat_e'] - 0.4) & 
    (res_df['int_l'] > res_df['int_e'] * 1.5) & 
    (res_df['act_l'] > res_df['act_e'] * 0.7)
].sort_values('sat_l')

# 2. 계속 좋아하는 곡 (Consistent): Sat High (>1.5) + Sat stable + Int short (<48h)
consistent = res_df[
    (res_df['sat_l'] > 1.5) & 
    (abs(res_df['sat_l'] - res_df['sat_e']) < 0.3) & 
    (res_df['int_l'] < 72)
].sort_values('plays', ascending=False)

# 3. 최근 더 좋아하게 된 곡 (Rising): Sat Increase (>0.3) + Int shortened
rising = res_df[
    (res_df['sat_l'] > res_df['sat_e'] + 0.3) & 
    (res_df['int_l'] < res_df['int_e'] * 0.8)
].sort_values('sat_l', ascending=False)

# 4. 최근 덜 좋아하게 된 곡 (Fading): Sat slightly drop or stable + Int Increase + Act maintained
fading = res_df[
    (res_df['sat_l'] <= res_df['sat_e']) & 
    (res_df['int_l'] > res_df['int_e'] * 2.0) & 
    (res_df['act_l'] > res_df['act_e'] * 0.7)
]
# Remove overlaps with burnout
fading = fading[~fading['song'].isin(burnout['song'])].sort_values('int_l', ascending=False)

def print_list(title, df_list):
    print(f"\n### {title}")
    if len(df_list) == 0:
        print("- 해당 없음")
        return
    for _, row in df_list.head(10).iterrows():
        # Shorten titles if too long
        title_disp = row['song'][:45] + "..." if len(row['song']) > 45 else row['song']
        print(f"- {title_disp}")
        print(f"  [지표] 만족도: {row['sat_e']:.1f}->{row['sat_l']:.1f} | 간격: {row['int_e']:.1f}h->{row['int_l']:.1f}h | (재생: {row['plays']}회)")

print("--- [사용자 데이터] 음악 청취 심리 체감 검증 분석 ---")
print_list("1. 확실히 질려버린 곡 (Burnout)", burnout)
print_list("2. 질리지 않는 소울 메이트 곡 (Consistent)", consistent)
print_list("3. 최근 스며들며 좋아진 곡 (Rising Fav)", rising)
print_list("4. 애정이 식어가는 중인 곡 (Fading Fav)", fading)
