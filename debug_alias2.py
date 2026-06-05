"""검정치마 매핑 로직 검증"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')
c = json.load(open(r'data\caches\lastfm_artist_sim_cache.json', 'r', encoding='utf-8'))

# 라이브러리 아티스트 중 캐시 키가 있는 것들
lib_with_cache = []
for k in c:
    if k.startswith('similar_artists||'):
        lib_with_cache.append(k.replace('similar_artists||', ''))

# '검정치마'가 유사 목록에 나오는 캐시 키들
appeared_in = []
for k, entries in c.items():
    if not k.startswith('similar_artists||'):
        continue
    src = k.replace('similar_artists||', '')
    for e in entries:
        if e['name'] == '검정치마':
            appeared_in.append(src)
            break

print(f"'검정치마'가 유사 목록에 나오는 캐시 키들 ({len(appeared_in)}개):")
for a in appeared_in:
    print(f"  {a}")

# The Black Skirts 유사 목록
bs_sims = c.get('similar_artists||The Black Skirts', [])
bs_sim_names = {e['name'].lower() for e in bs_sims}
overlap_bs = sum(1 for a in appeared_in if a.lower() in bs_sim_names)
print(f"\nThe Black Skirts 유사와 overlap: {overlap_bs}/{len(appeared_in)}")
print(f"  BS 유사: {[e['name'] for e in bs_sims[:10]]}")

# Jukjae 유사 목록
jk_sims = c.get('similar_artists||Jukjae', [])
if not jk_sims:
    jk_sims = c.get('similar_artists||jukjae', [])
jk_sim_names = {e['name'].lower() for e in jk_sims}
overlap_jk = sum(1 for a in appeared_in if a.lower() in jk_sim_names)
print(f"\nJukjae 유사와 overlap: {overlap_jk}/{len(appeared_in)}")
print(f"  Jukjae 유사: {[e['name'] for e in jk_sims[:10]]}")
