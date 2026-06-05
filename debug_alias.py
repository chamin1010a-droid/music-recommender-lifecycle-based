"""검정치마 매핑 디버그"""
import sys, os, io
sys.path.append(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\core')
sys.stdout.reconfigure(encoding='utf-8')
from lifecycle_recommender import run_pipeline

csv_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
meta_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\data\caches\ytm_metadata_cache.csv'
old = sys.stdout
sys.stdout = io.TextIOWrapper(open(os.devnull, 'wb'), encoding='utf-8')
r = run_pipeline(csv_path=csv_p, user_name='dbg', playlist_size=5, preset='default',
                 metadata_path=meta_p, user_birth_year=1998, skip_external=True)
sys.stdout = old
sys.stdout.reconfigure(encoding='utf-8')

mixer = r['mixer']
# 강제로 캐시 로드
sim_map = mixer._load_artist_sim_from_cache('JANNABI')
print("JANNABI sim_map에서 검정치마 관련:")
for k, v in sim_map.items():
    if 'black' in k or '검정' in k or 'skirt' in k:
        print(f"  {k}: {v}")

print("\n_alias_to_lib에서 검정치마 관련:")
for k, v in mixer._alias_to_lib.items():
    if 'black' in k or '검정' in k or 'skirt' in k:
        print(f"  {k} → {v}")

print("\n_get_artist_similarity('JANNABI', 'The Black Skirts'):")
print(f"  {mixer._get_artist_similarity('The Black Skirts', 'JANNABI')}")

# seed_artist_sim_cache 확인
mixer._seed_artist_sim_cache = {}
sim = mixer._get_artist_similarity('The Black Skirts', 'JANNABI')
print(f"  direct call: {sim}")

# 기본 캐시에서 JANNABI→검정치마 확인
print(f"\nJANNABI sim_map 전체 (상위10):")
items = sorted(sim_map.items(), key=lambda x: x[1], reverse=True)[:10]
for k, v in items:
    print(f"  {k}: {v}")
