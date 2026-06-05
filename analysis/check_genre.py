import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

cache_path = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\ytm_metadata_cache.csv'

try:
    df = pd.read_csv(cache_path, encoding='utf-8-sig')
    
    matched = df[df['matched'] == True]
    unmatched = df[df['matched'] == False]
    
    print("="*60)
    print("🎵 장르(메타데이터) 매칭 성공한 곡들 (Top 30)")
    print("="*60)
    for _, row in matched.head(30).iterrows():
        print(f"[✅ 성공] 아티스트: {row.get('itunes_artist', 'Unknown')[:15]:<15} | 노래: {row.get('itunes_title', 'Unknown')[:25]:<25} | 장르: {row.get('genre', 'Unknown')}")
        
    print("\n" + "="*60)
    print("💔 장르 매칭 실패/미수집 상태인 곡들 (Top 20)")
    print("="*60)
    for _, row in unmatched.head(20).iterrows():
        print(f"[❌ 실패] 입력값 -> 아티스트: {row['query_artist'][:15]:<15} | 노래: {row['query_title'][:25]:<25}")
        
    print(f"\n[통계] 성공한 곡 수: {len(matched)}곡 / 실패 및 캐싱 전: {len(unmatched)}곡")
    
except Exception as e:
    print(f"오류 발생: {e}")
