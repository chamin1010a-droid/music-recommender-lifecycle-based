import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

def check_top(csv_path, user_name):
    print(f"\n[{user_name} - 최애곡 월별 재생 횟수 추이]")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['timestamp'])
    
    top_songs = df['title'].value_counts().head(3).index
    df['month'] = df['date'].dt.to_period('M')
    
    for s in top_songs:
        print(f"\n곡명: {s}")
        print(df[df['title'] == s]['month'].value_counts().sort_index())

check_top(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\유튜브 뮤직 로그들\친구D\친구D_features.csv', '친구D')
check_top(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\유튜브 뮤직 로그들\친구B\친구B_features.csv', '친구B')
