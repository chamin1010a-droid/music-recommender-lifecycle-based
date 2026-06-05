import sys, os, io
sys.path.append(os.path.join(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트', 'core'))
sys.stdout.reconfigure(encoding='utf-8')
from lifecycle_recommender import run_pipeline

csv_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
meta_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\data\caches\ytm_metadata_cache.csv'

old = sys.stdout
sys.stdout = io.TextIOWrapper(open(os.devnull, 'wb'), encoding='utf-8')
r = run_pipeline(csv_path=csv_p, user_name='d', playlist_size=20, preset='default', metadata_path=meta_p, user_birth_year=1998)
sys.stdout = old
sys.stdout.reconfigure(encoding='utf-8')

sc = r['scorer']

# 찰리푸스 전곡을 제목에 키워드가 포함된 것들로 그룹핑
keywords = ['Hero', 'The Way I Am', 'Done for Me', 'How Long', 'Marvin Gaye', 'Left and Right', 'Light Switch', 'Attention']

print("찰리푸스 곡 버전별 점수 확인")
print("=" * 100)

for kw in keywords:
    print(f"\n🔍 '{kw}' 검색 결과:")
    matches = []
    for k, v in sc.song_scores.items():
        if v.get('artist','') == 'Charlie Puth - Topic' and kw.lower() in v.get('title','').lower():
            matches.append(v)
    
    matches.sort(key=lambda x: x.get('total_plays', 0), reverse=True)
    for m in matches:
        t = m['title'][:55]
        print(f"   {t:<55} 재생={m['total_plays']:>3}회 스킵={m['skip_rate']*100:>3.0f}% 호감={m['affinity']:.2f} 모멘={m['momentum']:.2f} 총합={m['affinity']*m['momentum']:.3f} last={m['days_since_last']}일전")
