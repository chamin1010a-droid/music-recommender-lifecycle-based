import json, sys
sys.stdout.reconfigure(encoding='utf-8')
c = json.load(open(r'data\caches\lastfm_artist_sim_cache.json', 'r', encoding='utf-8'))

artists = {
    'JANNABI': 'JANNABI',
    'The Black Skirts': 'The Black Skirts',
    'MC the MAX': 'MC the MAX',
    'SG Wannabe': 'SG Wannabe',
    'DAY6': 'DAY6',
}

for display, key in artists.items():
    cache_key = f"similar_artists||{key}"
    entries = c.get(cache_key, [])
    print(f"\n{'='*60}")
    print(f"🎵 {display} — 유사 아티스트 (상위 15)")
    print(f"{'='*60}")
    if not entries:
        print("  ❌ 캐시 없음!")
        continue
    for i, e in enumerate(entries[:15]):
        bar = '█' * int(e['match'] * 20)
        print(f"  {i+1:2d}. {e['name']:25s} {e['match']:.3f} {bar}")
