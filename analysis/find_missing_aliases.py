import pandas as pd
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

history_csv = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
df = pd.read_csv(history_csv, encoding='utf-8-sig')

top_songs = df.groupby(['artist', 'title']).size().reset_index(name='plays').sort_values('plays', ascending=False)

def has_korean(text):
    return bool(re.search('[가-힣]', str(text)))

missing_korean = top_songs[~top_songs['title'].apply(has_korean)]
print("=== 많이 들은 곡 중 한글 패치 안 된 곡들 ===")
count = 0
for idx, row in missing_korean.iterrows():
    if count >= 30:
        break
    artist = str(row['artist']).replace(' - Topic', '')
    if artist in ['JANNABI', 'The Black Skirts', 'Xdinary Heroes', 'DAY6', 'CNBLUE', 'M.C the MAX', 'Car, the Garden']:
        print(f"{artist:<20} | {row['title']:<40} | {row['plays']}회")
        count += 1
