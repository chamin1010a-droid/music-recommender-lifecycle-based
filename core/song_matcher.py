"""
[곡 매칭 유틸리티 — Song Matcher]

외부(Last.fm 등)에서 가져온 곡 제목과 내 라이브러리의 곡 제목을 
정확하게 매칭하기 위한 정규화 및 퍼지 매칭 엔진.

문제: "Good Night Good Dream" vs "‎Good Night Good Dream (좋은 밤 좋은 꿈)"
        → 괄호만 벗기면 됨

문제: "Nerd Connection" vs "Nerd Connection - Topic"
        → " - Topic" 제거하면 됨

문제: "검정치마" vs "The Black Skirts"
        → 알려진 alias 매핑 사용

이 모듈은 external_discovery.py에서 import하여 사용합니다.
"""

import re
from difflib import SequenceMatcher


def normalize_title(title: str) -> str:
    """
    곡 제목을 정규화하여 비교 가능한 형태로 변환.
    
    변환 규칙:
    1. 소문자로 통일
    2. 괄호 안 내용 제거: "좋은 밤 좋은 꿈 (Good Night Good Dream)" → "좋은 밤 좋은 꿈"
    3. 특수문자 제거 (하이픈, 점 등)
    4. 앞뒤 공백 제거
    5. 비가시 유니코드(LRM 등) 제거
    """
    t = str(title)
    # 비가시 유니코드 문자 제거 (예: \u200f, \u200e 등)
    t = re.sub(r'[\u200b-\u200f\u2028-\u202f\ufeff]', '', t)
    t = t.lower().strip()
    # 괄호 안 내용 + 괄호 제거
    t = re.sub(r'\s*\([^)]*\)', '', t)
    t = re.sub(r'\s*\[[^\]]*\]', '', t)
    # feat. 등 제거
    t = re.sub(r'\s*feat\.?\s*.*', '', t)
    # 특수문자 정리
    t = re.sub(r'[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ]', '', t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()


def extract_parenthesized(title: str) -> list:
    """
    제목에서 괄호 안의 내용을 모두 추출.
    "좋은 밤 좋은 꿈 (Good Night Good Dream)" → ["Good Night Good Dream"]
    """
    return re.findall(r'\(([^)]+)\)', str(title))


def normalize_artist(artist: str) -> str:
    """아티스트명 정규화"""
    a = str(artist)
    a = re.sub(r'[\u200b-\u200f\u2028-\u202f\ufeff]', '', a)
    a = a.replace(' - Topic', '').strip().lower()
    return a


# 알려진 아티스트 영문↔한글 매핑
ARTIST_ALIASES = {
    'the black skirts': '검정치마', '검정치마': 'the black skirts',
    'jannabi': '잔나비', '잔나비': 'jannabi',
    'hyukoh': '혁오', '혁오': 'hyukoh',
    'car, the garden': '카더가든', '카더가든': 'car, the garden',
    'car the garden': '카더가든',
    'damons year': '데이먼스 이어', '데이먼스 이어': 'damons year',
    '데이먼스 이어 damons year': 'damons year',
    'band nah': '밴드나', '밴드나': 'band nah',
    'nerd connection': '너드커넥션', '너드커넥션': 'nerd connection',
    'the volunteers': '더 볼런티어스', '더 볼런티어스': 'the volunteers',
    'daybreak': '데이브레이크', '데이브레이크': 'daybreak',
    'standing egg': '스탠딩 에그', '스탠딩 에그': 'standing egg',
    'nell': '넬', '넬': 'nell',
    'busker busker': '버스커 버스커', '버스커 버스커': 'busker busker',
    'o3ohn': '오존',
    'silica gel': '실리카겔', '실리카겔': 'silica gel',
    '10cm': '10cm', '십센치': '10cm',
    'mc the max': 'mc the max', 'm.c the max': 'mc the max',
    'park hyo shin': '박효신', '박효신': 'park hyo shin',
    'iu': '아이유', '아이유': 'iu',
}


def get_artist_aliases(artist: str) -> set:
    """주어진 아티스트의 모든 알려진 이름을 반환"""
    norm = normalize_artist(artist)
    aliases = {norm}
    if norm in ARTIST_ALIASES:
        alt = ARTIST_ALIASES[norm].lower()
        aliases.add(alt)
        # 재귀적으로 한 번 더 (A→B→C 체인)
        if alt in ARTIST_ALIASES:
            aliases.add(ARTIST_ALIASES[alt].lower())
    return aliases


class SongMatcher:
    """
    내 라이브러리의 곡과 외부 곡을 정확하게 매칭하는 엔진.
    
    매칭 전략 (순서대로 시도):
    1. 정규화된 제목 + 아티스트 직접 매칭
    2. 괄호 안 내용과 매칭 (Last.fm 영문 제목이 괄호 안에 있는 경우)
    3. 퍼지 매칭 (유사도 85% 이상)
    """
    
    def __init__(self, song_temps: dict):
        """
        song_temps에서 매칭용 인덱스를 구축.
        """
        self.song_temps = song_temps
        # 인덱스 1: {정규화된_아티스트: [{norm_title, orig_title, song_id, paren_titles}, ...]}
        self.artist_songs = {}
        
        for sid, info in song_temps.items():
            artist = str(info.get('artist', ''))
            title = str(info.get('title', ''))
            
            norm_artist = normalize_artist(artist)
            norm_title = normalize_title(title)
            paren_titles = [normalize_title(p) for p in extract_parenthesized(title)]
            
            entry = {
                'song_id': sid,
                'norm_title': norm_title,
                'orig_title': title,
                'paren_titles': paren_titles,
                'total_plays': info.get('total_plays', 0),
            }
            
            # 메인 아티스트명으로 등록
            for alias in get_artist_aliases(artist):
                if alias and alias != 'nan':
                    if alias not in self.artist_songs:
                        self.artist_songs[alias] = []
                    self.artist_songs[alias].append(entry)
    
    def is_in_library(self, artist: str, track: str) -> bool:
        """외부에서 가져온 (아티스트, 곡명)이 내 라이브러리에 있는지 확인"""
        norm_artist = normalize_artist(artist)
        norm_track = normalize_title(track)
        
        # 아티스트의 모든 alias로 탐색
        for alias in get_artist_aliases(artist):
            songs = self.artist_songs.get(alias, [])
            for s in songs:
                # 전략 1: 정규화된 제목 직접 비교
                if norm_track == s['norm_title']:
                    return True
                
                # 전략 2: 괄호 안 내용과 비교
                for pt in s['paren_titles']:
                    if norm_track == pt:
                        return True
                    # 괄호 내용이 외부 제목을 포함하거나 반대
                    if norm_track in pt or pt in norm_track:
                        if min(len(norm_track), len(pt)) > 3:
                            return True
                
                # 전략 3: 퍼지 매칭 (85% 이상)
                ratio = SequenceMatcher(None, norm_track, s['norm_title']).ratio()
                if ratio >= 0.85:
                    return True
        
        return False
    
    def get_artist_song_count(self, artist: str) -> int:
        """아티스트의 라이브러리 내 전체 보유 곡 수"""
        count = 0
        for alias in get_artist_aliases(artist):
            count += len(self.artist_songs.get(alias, []))
        return count
    
    def get_artist_known_count(self, artist: str, min_plays=5) -> int:
        """아티스트의 '진짜 아는 곡' 수 (min_plays회 이상 재생한 곡만)"""
        count = 0
        for alias in get_artist_aliases(artist):
            songs = self.artist_songs.get(alias, [])
            count += sum(1 for s in songs if s.get('total_plays', 0) >= min_plays)
        return count
