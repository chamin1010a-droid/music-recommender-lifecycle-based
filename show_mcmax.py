import json, sys
sys.stdout.reconfigure(encoding='utf-8')
c = json.load(open(r'data\caches\lastfm_artist_sim_cache.json', 'r', encoding='utf-8'))
sims = c.get('similar_artists||M.C the MAX', [])
print("🎵 M.C the MAX — 유사 아티스트 (상위 15)")
print("=" * 60)
for i, e in enumerate(sims[:15]):
    bar = '█' * int(e['match'] * 20)
    print(f"  {i+1:2d}. {e['name']:25s} {e['match']:.3f} {bar}")
