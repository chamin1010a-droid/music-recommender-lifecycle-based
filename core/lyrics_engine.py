"""
[가사 기반 유사도 엔진]
Genius API로 가사를 수집하고, sentence-transformers 다국어 모델로
임베딩하여 곡 간 감성/주제 유사도를 계산합니다.

모델: paraphrase-multilingual-MiniLM-L12-v2 (한국어+영어 동시 지원, 384차원)
"""

import os
import json
import re
import time
import numpy as np

class LyricsEngine:
    def __init__(self, genius_token=None, cache_file='lyrics_cache.json', embeddings_file='lyrics_embeddings.json'):
        self.genius_token = genius_token
        self.genius = None
        
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'caches')
        os.makedirs(cache_dir, exist_ok=True)
        
        self.cache_file = os.path.join(cache_dir, cache_file)
        self.embeddings_file = os.path.join(cache_dir, embeddings_file)
        
        self.lyrics_cache = {}   # song_id → lyrics text
        self.embeddings = {}     # song_id → 384-dim vector
        self._model = None
        
        self._embedding_matrix = None
        self._song_id_to_idx = {}
        
        self._load_caches()
    
    def _load_caches(self):
        """캐시 로드"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.lyrics_cache = json.load(f)
                print(f"  [가사 엔진] 가사 캐시 로드: {len(self.lyrics_cache)}곡")
            except:
                self.lyrics_cache = {}
        
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                self.embeddings = {k: np.array(v) for k, v in raw.items()}
                print(f"  [가사 엔진] 임베딩 캐시 로드: {len(self.embeddings)}곡")
            except:
                self.embeddings = {}
    
    def _save_lyrics_cache(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.lyrics_cache, f, ensure_ascii=False, indent=1)
    
    def _save_embeddings_cache(self):
        raw = {k: v.tolist() for k, v in self.embeddings.items()}
        with open(self.embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(raw, f)
    
    def _init_genius(self):
        """Genius 클라이언트 초기화"""
        if self.genius is None and self.genius_token:
            try:
                import lyricsgenius
                self.genius = lyricsgenius.Genius(
                    self.genius_token,
                    verbose=False,
                    remove_section_headers=True,
                    skip_non_songs=True,
                    timeout=10,
                    retries=2
                )
            except ImportError:
                print("  [가사 엔진] lyricsgenius 설치 필요: pip install lyricsgenius")
    
    def _init_model(self):
        """Sentence-Transformers 모델 초기화 (최초 1회)"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print("  [가사 엔진] 다국어 임베딩 모델 로딩 중...")
                self._model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                print("  [가사 엔진] 모델 준비 완료!")
            except ImportError:
                print("  [가사 엔진] sentence-transformers 설치 필요")
    
    def _clean_artist(self, artist):
        """아티스트 이름 정제"""
        clean = str(artist).replace(' - Topic', '').replace('VEVO', '').strip()
        return clean
    
    def _clean_title(self, title):
        """제목 정제 (괄호, 피처링 등 제거)"""
        clean = str(title)
        clean = re.sub(r'\s*\([^)]*\)', '', clean)  # (feat. xxx) 제거
        clean = re.sub(r'\s*\[[^\]]*\]', '', clean)  # [xxx] 제거
        clean = clean.strip()
        return clean
    
    def _artist_match(self, query_artist, result_artist):
        """Genius가 돌려준 아티스트가 우리가 검색한 아티스트와 같은지 검증"""
        from difflib import SequenceMatcher
        
        q = query_artist.lower().strip()
        r = result_artist.lower().strip()
        
        # 완전 일치
        if q == r:
            return True
        
        # 한쪽이 다른쪽에 포함
        if q in r or r in q:
            return True
        
        # 유사도 비교 (0.5 이상이면 통과)
        ratio = SequenceMatcher(None, q, r).ratio()
        if ratio >= 0.5:
            return True
        
        # 단어 단위 교집합 (아티스트명에 공통 단어가 1개라도 있으면)
        q_words = set(q.replace(',', ' ').split())
        r_words = set(r.replace(',', ' ').split())
        if q_words & r_words:
            return True
        
        return False
    
    def fetch_lyrics(self, artist, title, song_id=None):
        """
        Genius에서 가사 수집.
        검증: Genius가 돌려준 아티스트가 원본과 매칭되는지 확인.
        Returns: 가사 텍스트 또는 None
        """
        # 캐시 확인
        cache_key = song_id or f"{artist}||{title}"
        if cache_key in self.lyrics_cache:
            return self.lyrics_cache[cache_key]
        
        self._init_genius()
        if not self.genius:
            return None
        
        clean_artist = self._clean_artist(artist)
        clean_title = self._clean_title(title)
        
        try:
            song = self.genius.search_song(clean_title, clean_artist)
            if song and song.lyrics:
                # === 아티스트 검증 ===
                genius_artist = song.artist or ''
                if not self._artist_match(clean_artist, genius_artist):
                    # 오매칭: 가사 버림
                    self.lyrics_cache[cache_key] = None
                    return None
                
                # 가사 정제
                lyrics = song.lyrics
                # "123 ContributorsXXX Lyrics" 같은 헤더 제거
                lyrics = re.sub(r'^\d+\s*Contributors.*?Lyrics\s*', '', lyrics)
                # "Embed" 등 푸터 제거
                lyrics = re.sub(r'\d*Embed$', '', lyrics)
                lyrics = lyrics.strip()
                
                if len(lyrics) > 50:  # 최소 길이 체크
                    self.lyrics_cache[cache_key] = lyrics
                    return lyrics
            
            # 못 찾은 경우도 캐시 (재시도 방지)
            self.lyrics_cache[cache_key] = None
            return None
            
        except Exception as e:
            self.lyrics_cache[cache_key] = None
            return None
    
    def build_embeddings(self, song_dict, batch_save_interval=100):
        """
        전체 곡의 가사를 수집하고 임베딩 생성.
        song_dict: {song_id: {title, artist, ...}}
        """
        self._init_model()
        if not self._model:
            print("  [가사 엔진] 모델 로딩 실패, 중단")
            return
        
        # 1단계: 가사 수집
        to_fetch = [
            sid for sid in song_dict.keys()
            if sid not in self.lyrics_cache
        ]
        
        if to_fetch and self.genius_token:
            print(f"\n[가사 엔진] 가사 수집 시작: {len(to_fetch)}곡...")
            for i, sid in enumerate(to_fetch):
                info = song_dict[sid]
                self.fetch_lyrics(
                    info.get('artist', ''),
                    info.get('title', ''),
                    song_id=sid
                )
                
                if (i + 1) % batch_save_interval == 0:
                    found = sum(1 for v in self.lyrics_cache.values() if v)
                    print(f"  가사 수집 진행: {i+1}/{len(to_fetch)} (찾음: {found}곡)")
                    self._save_lyrics_cache()
                
                time.sleep(0.5)  # Rate limit 보호
            
            self._save_lyrics_cache()
            found = sum(1 for v in self.lyrics_cache.values() if v)
            print(f"  가사 수집 완료: {found}/{len(self.lyrics_cache)}곡 성공")
        
        # 2단계: 임베딩 생성
        to_embed = [
            sid for sid in song_dict.keys()
            if sid not in self.embeddings and self.lyrics_cache.get(sid)
        ]
        
        if to_embed:
            print(f"\n[가사 엔진] 임베딩 생성: {len(to_embed)}곡...")
            
            # 배치 처리 (효율)
            batch_size = 64
            for start in range(0, len(to_embed), batch_size):
                batch_ids = to_embed[start:start + batch_size]
                batch_lyrics = [self.lyrics_cache[sid] for sid in batch_ids]
                
                # 가사가 너무 길면 앞부분만 사용 (모델 입력 제한)
                batch_lyrics = [l[:2000] if l else "" for l in batch_lyrics]
                
                embeddings = self._model.encode(batch_lyrics, show_progress_bar=False)
                
                for sid, emb in zip(batch_ids, embeddings):
                    self.embeddings[sid] = emb
                
                if (start + batch_size) % (batch_size * 5) == 0:
                    print(f"  임베딩 진행: {min(start + batch_size, len(to_embed))}/{len(to_embed)}")
            
            self._save_embeddings_cache()
            print(f"  임베딩 완료: {len(self.embeddings)}곡")
        
        self._build_matrix()
    
    def _build_matrix(self):
        """전체 임베딩을 행렬로 구성"""
        if not self.embeddings:
            return
        
        song_ids = list(self.embeddings.keys())
        self._embedding_matrix = np.array([self.embeddings[sid] for sid in song_ids])
        self._song_id_to_idx = {sid: i for i, sid in enumerate(song_ids)}
    
    def calculate_similarity(self, song_a_id, song_b_id):
        """두 곡 간의 가사 유사도 (코사인, 0~1)"""
        if self._embedding_matrix is None:
            return None
        
        idx_a = self._song_id_to_idx.get(song_a_id)
        idx_b = self._song_id_to_idx.get(song_b_id)
        
        if idx_a is None or idx_b is None:
            return None
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        vec_a = self._embedding_matrix[idx_a].reshape(1, -1)
        vec_b = self._embedding_matrix[idx_b].reshape(1, -1)
        
        sim = cosine_similarity(vec_a, vec_b)[0][0]
        return float(max(0, sim))
    
    def calculate_similarity_to_vector(self, song_id, target_vector):
        """곡과 주어진 벡터 간의 유사도"""
        if self._embedding_matrix is None:
            return None
        
        idx = self._song_id_to_idx.get(song_id)
        if idx is None:
            return None
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        vec = self._embedding_matrix[idx].reshape(1, -1)
        sim = cosine_similarity(vec, target_vector.reshape(1, -1))[0][0]
        return float(max(0, sim))
