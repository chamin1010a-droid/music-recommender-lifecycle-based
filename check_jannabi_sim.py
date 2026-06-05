import sys,json
sys.stdout.reconfigure(encoding='utf-8')
c = json.load(open(r'data\caches\lastfm_artist_sim_cache.json','r',encoding='utf-8'))
sims = c.get('similar_artists||JANNABI',[])
print('JANNABI 유사 아티스트:')
for s in sims[:15]:
    name = s['name']
    match = s['match']
    print(f"  {name:30s} {match:.3f}")
