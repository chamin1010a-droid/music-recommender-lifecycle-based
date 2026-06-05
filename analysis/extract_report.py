import pandas as pd
import codecs, sys

sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

df = pd.read_csv(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv')

# Only consider valid plays (Not skipped within 30s)
valid_df = df[df['is_skipped'] == 0].copy()

print("=== USER REPORT ===")
print("TOTAL VALID PLAYS:", len(valid_df))

print("\n1. TOP 3 SONGS:")
print(valid_df['song_id'].value_counts().head(3))

print("\n2. TOP 5 ARTISTS:")
print(valid_df['artist'].value_counts().head(5))

print("\n3. DAY OF WEEK:")
valid_df['timestamp'] = pd.to_datetime(valid_df['timestamp'])
days = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
valid_df['day'] = valid_df['timestamp'].dt.dayofweek
for d, count in valid_df['day'].value_counts().head(3).items():
    print(f"{days[d]}: {count}")

print("\n4. TIME OF DAY (HOURS):")
print(valid_df['timestamp'].dt.hour.value_counts().head(3))

print("\n5. NEW VS OLD RATIO:")
new_count = len(valid_df[valid_df['familiarity'] == 0])
old_count = len(valid_df[valid_df['familiarity'] > 0])
total = new_count + old_count
print(f"NEW (Familiarity=0): {new_count / total * 100:.1f}%")
print(f"OLD (Familiarity>0): {old_count / total * 100:.1f}%")

