import requests
import json
import os
import time

class LastFMClient:
    """Last.fm API를 통한 곡/아티스트 태그 수집 및 유사 데이터 추출 클라이언트"""
    
    def __init__(self, api_key, cache_file='lastfm_cache.json'):
        self.api_key = api_key
        self.base_url = "http://ws.audioscrobbler.com/2.0/"
        # User-Agent 설정이 필수 (API Band 당하지 않기 위함)
        self.headers = {"User-Agent": "ChaminMusicProject/1.0 (chamin@example.com)"}
        
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _make_request(self, params):
        """API 호출 공통 함수. 성공 시 json 리턴, 실패 시 None 리턴"""
        params['api_key'] = self.api_key
        params['format'] = 'json'
        
        try:
            time.sleep(0.1) # 속도 제한 (초당 5회 보호)
            response = requests.get(self.base_url, params=params, headers=self.headers, timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"[LastFM Error] {e}")
            return None

    def get_track_tags(self, artist, track):
        """곡의 태그들을 가져온다 (예: ['indie rock', 'korean indie', 'melancholy'])"""
        cache_key = f"track_tags||{artist}||{track}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        data = self._make_request({
            'method': 'track.gettoptags',
            'artist': artist,
            'track': track,
            'autocorrect': 1
        })

        tags = []
        if data and 'toptags' in data and 'tag' in data['toptags']:
            # 리스트 형태일수도, 하나의 객체일수도 있음. 보통 리스트.
            tag_list = data['toptags']['tag']
            if isinstance(tag_list, dict):
                tag_list = [tag_list]
                
            # 노이즈를 줄이기 위해 카운트(가중치)가 어느정도 있는(count >= 5) 상위 태그만 추출 (최대 10개)
            for t in tag_list:
                if int(t.get('count', 0)) >= 5:
                    tags.append(t['name'].lower())
            
            tags = list(set(tags))[:10]
        
        self.cache[cache_key] = tags
        self._save_cache()
        return tags

    def get_artist_tags(self, artist):
        """아티스트의 태그들을 가져온다 (곡 태그가 비어있을 때 백업용)"""
        cache_key = f"artist_tags||{artist}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        data = self._make_request({
            'method': 'artist.gettoptags',
            'artist': artist,
            'autocorrect': 1
        })

        tags = []
        if data and 'toptags' in data and 'tag' in data['toptags']:
            tag_list = data['toptags']['tag']
            if isinstance(tag_list, dict):
                tag_list = [tag_list]
                
            for t in tag_list:
                if int(t.get('count', 0)) >= 10:
                    tags.append(t['name'].lower())
            tags = list(set(tags))[:15]

        self.cache[cache_key] = tags
        self._save_cache()
        return tags
    
    def get_combined_tags(self, artist, track):
        """
        곡의 태그와 아티스트 태그를 혼합하여 풍부한 특성 추출
        곡: "전설", 아티스트: "잔나비" -> 곡 태그 중심 + 아티스트 태그 덧붙임
        """
        # " - Topic" 등의 찌꺼기가 검색 무효화를 막도록 전처리
        clean_artist = str(artist).replace(" - Topic", "").strip()
        # 괄호 제거 (feat 등)
        import re
        clean_track = re.sub(r'\s*\([^)]*\)', '', str(track)).strip()

        track_tags = self.get_track_tags(clean_artist, clean_track)
        if not track_tags:
            # 곡 태그 검색 실패 시 아티스트 태그로 대체
            artist_tags = self.get_artist_tags(clean_artist)
            return artist_tags
        
        # 곡 태그가 있으면 거기에 아티스트 태그 중 일부만 추가
        artist_tags = self.get_artist_tags(clean_artist)
        combined = track_tags.copy()
        for t in artist_tags:
            if t not in combined:
                combined.append(t)
        return combined[:15]

    def get_similar_artists(self, artist, limit=20):
        """아티스트와 유사한 가수 목록 반환 (Last.fm의 크라우드소싱 데이터 기반)"""
        clean_artist = str(artist).replace(" - Topic", "").strip()
        cache_key = f"similar_artists||{clean_artist}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        data = self._make_request({
            'method': 'artist.getsimilar',
            'artist': clean_artist,
            'autocorrect': 1,
            'limit': limit
        })

        results = []
        if data and 'similarartists' in data and 'artist' in data['similarartists']:
            artist_list = data['similarartists']['artist']
            if isinstance(artist_list, dict):
                artist_list = [artist_list]
            for a in artist_list:
                results.append({
                    'name': a.get('name', ''),
                    'match': float(a.get('match', 0)),  # 유사도 0~1
                })

        self.cache[cache_key] = results
        self._save_cache()
        return results

    def get_artist_top_tracks(self, artist, limit=5):
        """아티스트의 대표곡(인기순) 반환"""
        clean_artist = str(artist).replace(" - Topic", "").strip()
        cache_key = f"top_tracks||{clean_artist}||{limit}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        data = self._make_request({
            'method': 'artist.gettoptracks',
            'artist': clean_artist,
            'autocorrect': 1,
            'limit': limit
        })

        results = []
        if data and 'toptracks' in data and 'track' in data['toptracks']:
            track_list = data['toptracks']['track']
            if isinstance(track_list, dict):
                track_list = [track_list]
            for t in track_list:
                results.append({
                    'name': t.get('name', ''),
                    'artist': t.get('artist', {}).get('name', ''),
                    'playcount': int(t.get('playcount', 0)),
                    'listeners': int(t.get('listeners', 0)),
                })

        self.cache[cache_key] = results
        self._save_cache()
        return results

    def get_tag_top_tracks(self, tag, limit=50):
        """특정 태그(장르)의 인기곡 목록 반환"""
        cache_key = f"tag_top_tracks||{tag.lower()}||{limit}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        data = self._make_request({
            'method': 'tag.gettoptracks',
            'tag': tag,
            'limit': limit
        })

        results = []
        if data and 'tracks' in data and 'track' in data['tracks']:
            track_list = data['tracks']['track']
            if isinstance(track_list, dict):
                track_list = [track_list]
            for t in track_list:
                results.append({
                    'name': t.get('name', ''),
                    'artist': t.get('artist', {}).get('name', ''),
                })

        self.cache[cache_key] = results
        self._save_cache()
        return results

    def get_geo_top_tracks(self, country='south korea', limit=50):
        """특정 국가의 인기곡 목록 반환"""
        cache_key = f"geo_top||{country}||{limit}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        data = self._make_request({
            'method': 'geo.gettoptracks',
            'country': country,
            'limit': limit
        })

        results = []
        if data and 'tracks' in data and 'track' in data['tracks']:
            track_list = data['tracks']['track']
            if isinstance(track_list, dict):
                track_list = [track_list]
            for t in track_list:
                results.append({
                    'name': t.get('name', ''),
                    'artist': t.get('artist', {}).get('name', ''),
                    'listeners': int(t.get('listeners', 0)),
                })

        self.cache[cache_key] = results
        self._save_cache()
        return results

