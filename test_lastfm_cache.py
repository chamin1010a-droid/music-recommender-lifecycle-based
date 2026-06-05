import os
"""캐싱 스크립트 빠른 테스트 (3명만)"""
import sys, os, time
sys.path.append(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\core')
sys.stdout.reconfigure(encoding='utf-8')
from lastfm_client import LastFMClient

LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY", "")
cache_file = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\data\caches\lastfm_artist_sim_cache.json'
os.makedirs(os.path.dirname(cache_file), exist_ok=True)

lastfm = LastFMClient(api_key=LASTFM_API_KEY, cache_file=cache_file)

test_artists = ['The Black Skirts', 'SG Wannabe', 'DAY6']
for artist in test_artists:
    results = lastfm.get_similar_artists(artist, limit=10)
    print(f"\n🎤 {artist}:")
    for r in results[:5]:
        print(f"  {r['name']:30s} match: {r['match']:.3f}")
    if not results:
        print("  (결과 없음)")

print(f"\n캐시 저장 위치: {cache_file}")
print(f"캐시 항목 수: {len(lastfm.cache)}")
