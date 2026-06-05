"""검정치마 별명 매핑 확인"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')
c = json.load(open(r'data\caches\lastfm_artist_sim_cache.json', 'r', encoding='utf-8'))

# JANNABI의 유사 목록에 '검정치마'가 있음
sims = c.get('similar_artists||JANNABI', [])
for s in sims[:3]:
    print(f"JANNABI 유사: {s['name']} ({s['match']})")

# The Black Skirts 캐시 키가 있는지?
key = 'similar_artists||The Black Skirts'
if key in c:
    bs_sims = c[key][:3]
    print(f"\nThe Black Skirts 캐시: {[s['name'] for s in bs_sims]}")
else:
    print(f"\n❌ '{key}' 캐시 키 없음!")
    # 그럼 뭐로 있어?
    for k in c:
        if 'black' in k.lower() or '검정' in k.lower():
            print(f"  발견: {k}")
