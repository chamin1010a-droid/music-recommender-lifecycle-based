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

# Show first 5 keys and their types
print("=== song_scores key type ===")
keys = list(sc.song_scores.keys())
print(f"Total songs: {len(keys)}")
print(f"Key type: {type(keys[0])}")
for k in keys[:5]:
    print(f"  Key: {repr(k)}")
    v = sc.song_scores[k]
    print(f"    title={v.get('title','?')}, artist={v.get('artist','?')}")

# Search by value
print("\n=== Searching by value fields ===")
for k, v in sc.song_scores.items():
    title = v.get('title', '')
    artist = v.get('artist', '')
    if 'Hero' == title and 'Charlie' in artist:
        print(f"FOUND Hero: key={repr(k)}, aff={v['affinity']}, mom={v['momentum']}")
    if 'Good Boy Twist' == title:
        print(f"FOUND GBT: key={repr(k)}, aff={v['affinity']}, mom={v['momentum']}")
    if 'Enemy' == title and 'Xdinary' in artist:
        print(f"FOUND Enemy: key={repr(k)}, aff={v['affinity']}, mom={v['momentum']}")
    if 'PLUTO' == title and 'Xdinary' in artist:
        print(f"FOUND PLUTO: key={repr(k)}, aff={v['affinity']}, mom={v['momentum']}")
    if 'Marvin Gaye' in title and 'Charlie' in artist:
        print(f"FOUND Marvin: key={repr(k)}, aff={v['affinity']}, mom={v['momentum']}, plays={v['total_plays']}")
