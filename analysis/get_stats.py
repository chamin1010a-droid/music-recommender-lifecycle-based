import pandas as pd
import sys

# Set stdout to utf-8
sys.stdout.reconfigure(encoding='utf-8')

csv_path = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\유튜브 뮤직 로그들\친구D\친구D_features.csv'
df = pd.read_csv(csv_path, encoding='utf-8-sig')

valid_df = df[df['is_skipped'] == 0]

print("--- TOP 5 SONGS ---")
print(valid_df['song_id'].value_counts().head(5))

print("\n--- TOP 1 ARTIST ---")
print(valid_df['artist'].value_counts().head(1))

print("\n--- MOST ACTIVE HOURS ---")
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
print(df['hour'].value_counts().head(3))
