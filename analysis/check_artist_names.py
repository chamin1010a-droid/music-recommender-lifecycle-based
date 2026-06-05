import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

csv_path = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
df = pd.read_csv(csv_path, encoding='utf-8-sig')

artists = df['artist'].unique()
topic = [a for a in artists if '- Topic' in str(a)]
non_topic = [a for a in artists if '- Topic' not in str(a) and str(a) != 'nan']

print(f'Topic 아티스트: {len(topic)}개')
print(f'Non-Topic 아티스트: {len(non_topic)}개\n')

print('Non-Topic 아티스트 (재생 횟수 상위 30):')
for a in sorted(non_topic, key=lambda x: len(df[df['artist']==x]), reverse=True)[:30]:
    cnt = len(df[df['artist']==a])
    # 같은 이름이 Topic에도 있는지 확인
    base = a.replace('VEVO', '').replace('vevo', '').strip()
    similar_topic = [t for t in topic if base.lower() in t.lower()]
    marker = f' ← 합칠 후보: {similar_topic[0]}' if similar_topic else ''
    print(f'  {cnt:>4}회 | {a}{marker}')
