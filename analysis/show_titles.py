import pandas as pd
import codecs, sys

sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

df = pd.read_csv(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv')

# 한국 아티스트들의 전체 제목 목록 출력
korean_artists = [
    'JANNABI', 'HANRORO', 'Car, the Garden', 'Damons Year',
    'M.C the MAX', 'Monday Kiz', 'SG Wannabe', 'Buzz', 'December',
    'DAY6', 'Xdinary Heroes', 'Choi Yu Ree', 'Sung Si-kyung',
    'Nerd Connection', 'Silica Gel', 'AKMU', 'NELL', 'Jang Beom June',
    'Broccoli', 'Park Hyo Shin', 'Release', 'CNBLUE', 'DAVICHI',
    'The Electriceels', 'The Volunteers', 'Huh Gak', 'Ha Hyun Sang',
    'SAM KIM', 'Deli Spice', 'The Black Skirts'
]

for artist in korean_artists:
    sub = df[df['artist'].str.contains(artist, na=False, case=False)]
    if len(sub) == 0:
        continue
    titles = sub['title'].value_counts()
    print(f"\n=== {artist} ({len(sub)}회) ===")
    for title, count in titles.items():
        print(f"  {title}: {count}회")
