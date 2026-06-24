"""
[음악 추천 웹 앱]
Flask 서버 — 추천 엔진 + YouTube Music 연동
"""

import sys
import os
import json

# .env 파일에서 환경변수 로드
from pathlib import Path
_env_path = Path(__file__).parent / '.env'
if _env_path.exists():
    for line in _env_path.read_text(encoding='utf-8').strip().splitlines():
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 프로젝트 경로
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(PROJECT_DIR, 'core')
sys.path.insert(0, CORE_DIR)

from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__, 
            template_folder=os.path.join(PROJECT_DIR, 'web', 'templates'),
            static_folder=os.path.join(PROJECT_DIR, 'web', 'static'))

# ── 데이터 로드 ──
MUSIC_USER = os.environ.get('MUSIC_USER', 'user')  # 로컬 .env에서 실제 데이터 폴더명 지정
CSV_PATH = os.path.join(PROJECT_DIR, '유튜브 뮤직 로그들', MUSIC_USER, f'{MUSIC_USER}_features.csv')
BROWSER_JSON = os.path.join(PROJECT_DIR, 'data', 'caches', 'browser.json')

df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
song_list = df.groupby('song_id').first().reset_index()
song_list = song_list[['song_id', 'title', 'artist']].copy()
song_list['artist'] = song_list['artist'].fillna('Unknown').astype(str).str.replace(' - Topic', '', regex=False)
song_list = song_list.sort_values('title').to_dict('records')

print(f"[서버] {len(song_list)}곡 로드 완료")


# ── 유사도: MERT 임베딩 (MFCC 대비 음악 이해 우수, 캐시 사전계산) ──
import numpy as np
print("[엔진] MERT 임베딩 로드 중...")
_mert = {}
try:
    with open(os.path.join(PROJECT_DIR, 'data', 'caches', 'audio_mert_cache.json'), 'r', encoding='utf-8') as f:
        _mert_raw = json.load(f)
    for _k, _v in _mert_raw.items():
        _a = np.asarray(_v, dtype=float)
        _mert[_k] = _a / (np.linalg.norm(_a) + 1e-9)
    print(f"[엔진] MERT {len(_mert)}곡(768D) 정규화 완료")
except Exception as e:
    print(f"[엔진] MERT 로드 실패: {e}")

# ── 호감도 점수 로드 ──
SCORES_PATH = os.path.join(PROJECT_DIR, 'data', 'caches', 'song_scores.json')
_song_scores = {}
try:
    with open(SCORES_PATH, 'r', encoding='utf-8') as f:
        _song_scores = json.load(f)
    print(f"[점수] {len(_song_scores)}곡 호감도 점수 로드 완료")
except Exception as e:
    print(f"[점수] 로드 실패 (export_scores.py를 먼저 실행하세요): {e}")
print()


# ── YouTube Music API ──
_ytmusic = None

OAUTH_JSON = os.path.join(PROJECT_DIR, 'data', 'caches', 'oauth.json')
OAUTH_CLIENT_ID = os.environ.get("YTM_CLIENT_ID", "")
OAUTH_CLIENT_SECRET = os.environ.get("YTM_CLIENT_SECRET", "")
BROWSER_JSON = os.path.join(PROJECT_DIR, 'data', 'caches', 'browser.json')

_ytmusic_search = None   # 검색용 (인증 불필요)
_ytmusic_auth = None     # 등록용 (인증 필요)

def _refresh_oauth_token():
    """OAuth refresh_token으로 access_token 자동 갱신"""
    import requests as req
    import time as t
    oauth = json.load(open(OAUTH_JSON, encoding='utf-8'))
    resp = req.post('https://oauth2.googleapis.com/token', data={
        'client_id': OAUTH_CLIENT_ID,
        'client_secret': OAUTH_CLIENT_SECRET,
        'refresh_token': oauth['refresh_token'],
        'grant_type': 'refresh_token'
    })
    new_token = resp.json()
    if 'access_token' in new_token:
        oauth['access_token'] = new_token['access_token']
        oauth['expires_at'] = t.time() + new_token.get('expires_in', 3600)
        with open(OAUTH_JSON, 'w') as f:
            json.dump(oauth, f, indent=2)
        print("[OAuth] 토큰 갱신 완료")
        return oauth['access_token']
    return None

def get_ytmusic_search():
    """검색용 (인증 없음 — 무제한)"""
    global _ytmusic_search
    if _ytmusic_search is None:
        from ytmusicapi import YTMusic
        _ytmusic_search = YTMusic()
    return _ytmusic_search

def get_ytmusic_auth():
    """등록용 (OAuth 인증 — 자동 갱신)"""
    global _ytmusic_auth
    import time as t
    from ytmusicapi import YTMusic
    
    # 토큰 만료 체크 → 갱신
    try:
        oauth = json.load(open(OAUTH_JSON, encoding='utf-8'))
        if t.time() > oauth.get('expires_at', 0) - 300:
            _refresh_oauth_token()
    except:
        pass
    
    # browser.json 또는 OAuth 로 초기화
    if _ytmusic_auth is None:
        try:
            # browser.json이 있으면 우선 사용
            if os.path.exists(BROWSER_JSON):
                _ytmusic_auth = YTMusic(BROWSER_JSON)
                # 동작 확인
                _ytmusic_auth.get_library_playlists(limit=1)
                print("[YTM] browser.json 인증 활성")
            else:
                raise FileNotFoundError
        except:
            try:
                from ytmusicapi import OAuthCredentials
                _ytmusic_auth = YTMusic(OAUTH_JSON, oauth_credentials=OAuthCredentials(
                    client_id=OAUTH_CLIENT_ID, client_secret=OAUTH_CLIENT_SECRET
                ))
                print("[YTM] OAuth 인증 활성")
            except:
                print("[YTM] 인증 실패 — 플레이리스트 등록 불가")
                _ytmusic_auth = YTMusic()
    
    return _ytmusic_auth


# ── 라우트 ──

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/songs')
def api_songs():
    """전체 곡 목록 (검색용)"""
    q = request.args.get('q', '').lower()
    if q:
        filtered = [s for s in song_list 
                     if q in s['title'].lower() or q in s['artist'].lower()]
        return jsonify(filtered[:30])
    return jsonify(song_list[:50])


@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """시드곡 기반 추천 생성"""
    data = request.json
    seed_title = data.get('seed_title', '')
    seed_artist = data.get('seed_artist', '')
    count = data.get('count', 20)

    try:
        import numpy as np

        # 시드 song_id 찾기
        seed_id = None
        for s in song_list:
            if (seed_title.lower() in s['title'].lower() and
                seed_artist.lower()[:4] in s['artist'].lower()):
                seed_id = s['song_id']
                break
        if not seed_id:
            return jsonify({'error': f'시드곡을 찾을 수 없습니다: {seed_title}'}), 404
        if seed_id not in _mert:
            return jsonify({'error': f'시드곡 임베딩이 없습니다: {seed_title}'}), 404

        meta = {s['song_id']: s for s in song_list}
        seed_vec = _mert[seed_id]

        # 후보 = MERT 임베딩 + 점수(A) 둘 다 있는 곡, 유사도 = MERT 코사인
        cand = []
        for s in song_list:
            sid = s['song_id']
            if sid == seed_id or sid not in _mert or sid not in _song_scores:
                continue
            cand.append((sid, float(seed_vec @ _mert[sid])))
        cand.sort(key=lambda x: -x[1])
        gated = cand[:40]                                    # 느슨한 관문(상위 40 유사)
        simof = {sid: sim for sid, sim in gated}

        # 추천 점수 = 유사도 × 사랑(A).
        # ('생애주기'(질림·신선도·위치) 신호는 검증 결과 전부 진폭으로 환원돼 제거)
        score = lambda s: simof[s] * _song_scores[s]['A']
        ranked = sorted(simof, key=lambda s: -score(s))

        # 가중 랜덤: 상위권에서 점수에 비례해 뽑아 매번 조금씩 다른 리스트
        k = min(count - 1, len(ranked))
        pool = ranked[:max(k * 2, k)]
        w = np.clip(np.array([score(s) for s in pool], dtype=float), 1e-9, None)
        w = w / w.sum()
        pick = [pool[i] for i in np.random.choice(len(pool), size=k, replace=False, p=w)]
        pick = sorted(pick, key=lambda s: -score(s))

        # 시드곡을 1번으로, 이어서 점수순
        playlist = [{'song_id': seed_id, 'title': seed_title, 'artist': seed_artist,
                     'tag': '시드', 'similarity': 1.0}]
        for sid in pick:
            m = meta[sid]
            playlist.append({
                'song_id': sid, 'title': m['title'], 'artist': m['artist'],
                'tag': '추천', 'similarity': round(float(simof[sid]), 3)
            })

        return jsonify({
            'seed': {'title': seed_title, 'artist': seed_artist, 'song_id': seed_id},
            'playlist': playlist
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/search_ytmusic', methods=['POST'])
def api_search_ytmusic():
    """YouTube Music에서 곡 검색하여 videoId 매핑"""
    data = request.json
    songs = data.get('songs', [])
    
    yt = get_ytmusic_auth()
    results = []
    
    for song in songs:
        title = song.get('title', '')
        artist = song.get('artist', '')
        query = f"{title} {artist}"
        
        try:
            try:
                search = yt.search(query, filter='songs', limit=1)
            except:
                # 인증 만료 시 fallback
                yt_fallback = get_ytmusic_search()
                search = yt_fallback.search(query, filter='videos', limit=3)
            
            if search:
                r = search[0]
                results.append({
                    'original_title': title,
                    'original_artist': artist,
                    'ytm_title': r['title'],
                    'ytm_artist': r['artists'][0]['name'] if r.get('artists') else 'Unknown',
                    'videoId': r['videoId'],
                    'matched': True
                })
            else:
                results.append({
                    'original_title': title,
                    'original_artist': artist,
                    'matched': False
                })
        except Exception as e:
            results.append({
                'original_title': title,
                'original_artist': artist,
                'matched': False,
                'error': str(e)
            })
    
    return jsonify(results)


@app.route('/api/create_playlist', methods=['POST'])
def api_create_playlist():
    """YouTube Music에 플레이리스트 생성"""
    data = request.json
    name = data.get('name', '추천 플레이리스트')
    description = data.get('description', '음악 추천 엔진이 생성한 플레이리스트')
    video_ids = data.get('video_ids', [])
    
    if not video_ids:
        return jsonify({'error': '곡이 없습니다'}), 400
    
    try:
        yt = get_ytmusic_auth()
        playlist_id = yt.create_playlist(
            title=name,
            description=description,
            video_ids=video_ids,
            privacy_status='PRIVATE'
        )
        
        return jsonify({
            'success': True,
            'playlist_id': playlist_id,
            'url': f'https://music.youtube.com/playlist?list={playlist_id}',
            'count': len(video_ids)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*50)
    print("🎵 음악 추천 웹 앱")
    print("   http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
