import pandas as pd
import urllib.parse
import urllib.request
import json
import ssl
import time
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from concurrent.futures import ThreadPoolExecutor, as_completed

# SSL 검증 무시 설정 (itunes 서버 통신 시 가끔 발생하는 인증서 오류 방지)
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def fetch_itunes_metadata(song_id, artist, title):
    # ArtistNameNormalizer에서 '- Topic' 등이 붙어있다면 1차 정제
    clean_artist = str(artist).replace(' - Topic', '').replace('VEVO', '').strip()
    clean_title = str(title).strip()
    
    # 쿼리 생성
    query = f"{clean_artist} {clean_title}"
    url = f"https://itunes.apple.com/search?term={urllib.parse.quote(query)}&entity=song&limit=1"
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    result = {
        'song_id': song_id,
        'query_artist': clean_artist,
        'query_title': clean_title,
        'itunes_artist': None,
        'itunes_title': None,
        'release_date': None,
        'release_year': None,
        'genre': None,
        'matched': False
    }
    
    try:
        # iTunes API Rate limit 우회 및 부하 경감을 위한 작은 딜레이
        time.sleep(0.2)
        with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            if data['resultCount'] > 0:
                item = data['results'][0]
                result['itunes_artist'] = item.get('artistName')
                result['itunes_title'] = item.get('trackName')
                result['release_date'] = item.get('releaseDate')
                
                # '2019-03-13T12:00:00Z' 형식에서 연도 추출
                if result['release_date']:
                    try:
                        result['release_year'] = int(result['release_date'][:4])
                    except:
                        pass
                        
                result['genre'] = item.get('primaryGenreName')
                result['matched'] = True
    except Exception as e:
        # 타임아웃, 연결 오류 등 처리
        pass
        
    return result

def main():
    base_dir = r"c:\Users\user\Desktop\데이터분석\음악 프로젝트"
    history_csv = os.path.join(base_dir, "Takeout", "YouTube 및 YouTube Music", "시청 기록", "ytm_history_features.csv")
    cache_csv = os.path.join(base_dir, "ytm_metadata_cache.csv")
    
    # 데이터 로드
    print(f"📁 히스토리 로드 중: {history_csv}")
    df = pd.read_csv(history_csv, encoding='utf-8-sig')
    
    # 고유 곡 식별
    unique_songs = df[['song_id', 'artist', 'title']].drop_duplicates('song_id')
    print(f"총 고유 곡 수: {len(unique_songs)}개")
    
    # 기존 캐시 로드 (이어받기 위함)
    cached_ids = set()
    results = []
    if os.path.exists(cache_csv):
        cache_df = pd.read_csv(cache_csv, encoding='utf-8-sig')
        cached_ids = set(cache_df['song_id'].tolist())
        results = cache_df.to_dict('records')
        print(f"기존 캐시 발견: {len(cached_ids)}곡 스킵")
        
    # 수집 대상 곡 필터링
    tasks = []
    for _, row in unique_songs.iterrows():
        if row['song_id'] not in cached_ids or (row['song_id'] in cached_ids and next((c['matched'] for c in results if c['song_id'] == row['song_id']), True) == False):
            # 캐시에 없거나, 캐시에 있는데 matched가 False인 경우 재시도
            tasks.append((row['song_id'], row['artist'], row['title']))
            
    print(f"🚀 수집 시작: {len(tasks)}곡 대상 (안전 모드: 단일 스레드, 1초 딜레이)")
    
    # 캐시에서 실패한 건들은 삭제하고 시작
    results = [r for r in results if r['matched']]
    
    completed = 0
    
    for sid, art, tit in tasks:
        try:
            # 완전한 안전주의 딜레이 (애플 밴 방지)
            time.sleep(1.2)
            res = fetch_itunes_metadata(sid, art, tit)
            results.append(res)
        except Exception as exc:
            print(f"에러 발생: {exc}")
            
        completed += 1
        if completed % 20 == 0:
            print(f"진행 상황: ... {completed}/{len(tasks)} 곡 완료")
            pd.DataFrame(results).to_csv(cache_csv, index=False, encoding='utf-8-sig')
    
    # 최종 저장
    final_df = pd.DataFrame(results)
    final_df.to_csv(cache_csv, index=False, encoding='utf-8-sig')
    
    matched_count = final_df['matched'].sum()
    print("\n" + "="*50)
    print("✅ 데이터 수집 완료")
    print(f"총 데이터: {len(final_df)}개")
    print(f"매칭 성공: {matched_count}개 ({matched_count/len(final_df)*100:.1f}%)")
    print("="*50)

if __name__ == '__main__':
    main()
