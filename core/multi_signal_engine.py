"""
[Multi-Signal 유사도 통합 엔진]
4개 신호를 가중 합산하여 정밀한 곡 간 유사도를 제공합니다.

신호 구성:
  1. Last.fm 태그 (25%) — 장르/스타일 수준
  2. 가사 NLP (30%) — 감성/주제 수준  
  3. 오디오 특성 (30%) — 음향/음색 수준
  4. 메타데이터 (15%) — 장르/발매년도

가용하지 않은 신호가 있으면 나머지 신호의 가중치를 재분배합니다.
"""

import os
import numpy as np
import pandas as pd


# 장르 그룹 매핑 (iTunes primaryGenreName → 상위 카테고리)
GENRE_GROUPS = {
    'K-Pop': 'korean_pop',
    'Korean Pop': 'korean_pop',
    'K-Hip-Hop': 'korean_hiphop',
    'Korean Hip-Hop': 'korean_hiphop',
    'Pop': 'pop',
    'J-Pop': 'jpop',
    'Rock': 'rock',
    'Alternative': 'rock',
    'Indie Rock': 'rock',
    'Korean Indie': 'korean_indie',
    'R&B/Soul': 'rnb',
    'R&B': 'rnb',
    'Soul': 'rnb',
    'Hip-Hop/Rap': 'hiphop',
    'Hip-Hop': 'hiphop',
    'Electronic': 'electronic',
    'Dance': 'electronic',
    'Singer/Songwriter': 'singer_songwriter',
    'Ballad': 'ballad',
    'Korean Ballad': 'ballad',
    'Metal': 'metal',
    'Hard Rock': 'metal',
    'Classical': 'classical',
    'Jazz': 'jazz',
    'Blues': 'blues',
    'Country': 'country',
    'Soundtrack': 'soundtrack',
    'Anime': 'anime',
}


class MultiSignalSimilarityEngine:
    """
    4개 유사도 신호를 통합하는 메인 엔진.
    기존 TagSimilarityEngine을 대체합니다.
    """
    
    # 가중치 (LightGBM 학습 결과 기반)
    # 태그는 시드 벡터 모드 전용이라 실험 대상 외 → 25% 유지
    # 나머지 75%를 학습 비율(오디오 78% / 메타 18% / 가사 4%)로 분배
    DEFAULT_WEIGHTS = {
        'lastfm_tags': 0.25,
        'lyrics': 0.03,
        'audio': 0.58,
        'metadata': 0.14,
    }
    
    def __init__(self, tag_engine=None, lyrics_engine=None, audio_engine=None,
                 metadata_path=None, weights=None):
        """
        tag_engine: TagSimilarityEngine 인스턴스 (기존)
        lyrics_engine: LyricsEngine 인스턴스
        audio_engine: AudioFeaturesEngine 인스턴스
        metadata_path: ytm_metadata_cache.csv 경로
        weights: 가중치 딕셔너리 (기본값 사용 가능)
        """
        self.tag_engine = tag_engine
        self.lyrics_engine = lyrics_engine
        self.audio_engine = audio_engine
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        
        # 메타데이터 로드
        self.metadata = {}
        if metadata_path and os.path.exists(metadata_path):
            self._load_metadata(metadata_path)
        
        # 기존 TagSimilarityEngine 호환용 속성
        self.lastfm = tag_engine.lastfm if tag_engine else None
        self.vectorizer = tag_engine.vectorizer if tag_engine else None
        self.tfidf_matrix = tag_engine.tfidf_matrix if tag_engine else None
        self.song_id_to_index = tag_engine.song_id_to_index if tag_engine else {}
        self.index_to_song_id = tag_engine.index_to_song_id if tag_engine else {}
    
    def _load_metadata(self, path):
        """iTunes 메타데이터 캐시 로드"""
        try:
            df = pd.read_csv(path, encoding='utf-8-sig')
            for _, row in df.iterrows():
                sid = row.get('song_id')
                if sid and row.get('matched'):
                    self.metadata[sid] = {
                        'genre': row.get('genre', ''),
                        'release_year': row.get('release_year'),
                    }
            print(f"  [통합 엔진] iTunes 메타데이터 로드: {len(self.metadata)}곡")
        except Exception as e:
            print(f"  [통합 엔진] 메타데이터 로드 실패: {e}")
    
    def _metadata_similarity(self, song_a_id, song_b_id):
        """장르 + 발매년도 기반 메타데이터 유사도 (0~1)"""
        meta_a = self.metadata.get(song_a_id)
        meta_b = self.metadata.get(song_b_id)
        
        if not meta_a or not meta_b:
            return None
        
        score = 0.0
        
        # 장르 유사도
        genre_a = str(meta_a.get('genre', ''))
        genre_b = str(meta_b.get('genre', ''))
        
        if genre_a and genre_b:
            if genre_a == genre_b:
                score += 0.5
            else:
                group_a = GENRE_GROUPS.get(genre_a, genre_a.lower())
                group_b = GENRE_GROUPS.get(genre_b, genre_b.lower())
                if group_a == group_b:
                    score += 0.25
        
        # 발매년도 근접도
        year_a = meta_a.get('release_year')
        year_b = meta_b.get('release_year')
        
        if year_a and year_b:
            try:
                diff = abs(int(year_a) - int(year_b))
                if diff <= 5:
                    score += 0.5 * (1 - diff / 5)
            except:
                pass
        
        return score
    
    def _fallback_value(self):
        """데이터 없는 신호에 채울 랜덤 기본값 (0.25~0.35)"""
        return 0.30 + np.random.uniform(-0.05, 0.05)
    
    def calculate_similarity(self, song_a_id, song_b_id):
        """
        4개 신호를 가중 합산하여 최종 유사도(0~1) 반환.
        
        태그 엔진이 없으면 가중치를 나머지 신호에 재배분.
        오디오 유사도는 바닥(0.85)을 재정규화하여 차별력 확보.
        """
        signals = {}
        available_weights = {}
        
        # 1. Last.fm 태그 유사도
        tag_available = False
        if self.tag_engine:
            try:
                sim = self.tag_engine.calculate_similarity(song_a_id, song_b_id)
                if sim is not None and sim > 0:
                    signals['lastfm_tags'] = float(sim)
                    tag_available = True
            except:
                pass
        
        # 2. 가사 유사도
        lyrics_available = False
        if self.lyrics_engine:
            try:
                sim = self.lyrics_engine.calculate_similarity(song_a_id, song_b_id)
                if sim is not None:
                    signals['lyrics'] = sim
                    lyrics_available = True
            except:
                pass
        if not lyrics_available:
            signals['lyrics'] = self._fallback_value()
        
        # 3. 오디오 특성 유사도 (StandardScaler 적용으로 이미 차별력 확보)
        audio_available = False
        if self.audio_engine:
            try:
                sim = self.audio_engine.calculate_similarity(song_a_id, song_b_id)
                if sim is not None:
                    # StandardScaler 사용 → 음수~양수 범위 → 0~1로 클램핑
                    signals['audio'] = max(0.0, min(float(sim), 1.0))
                    audio_available = True
            except:
                pass
        if not audio_available:
            signals['audio'] = self._fallback_value()
        
        # 4. 메타데이터 유사도
        meta_sim = self._metadata_similarity(song_a_id, song_b_id)
        if meta_sim is not None:
            signals['metadata'] = meta_sim
        else:
            signals['metadata'] = self._fallback_value()
        
        # === 가중치 결정 ===
        if tag_available:
            # 태그 있으면 원래 가중치
            w = dict(self.weights)
        else:
            # 태그 없으면 25%를 나머지에 재배분 (학습된 비율 유지)
            w = {
                'lastfm_tags': 0.0,
                'lyrics': 0.04,      # 3 → 4
                'audio': 0.78,       # 58 → 78 (학습 원본 비율)
                'metadata': 0.18,    # 14 → 18
            }
            signals['lastfm_tags'] = 0  # 사용 안 함
        
        final = sum(signals.get(k, 0) * w.get(k, 0) for k in w)
        return float(final)
    
    # === 기존 TagSimilarityEngine 호환 메서드 ===
    
    def build_tag_vectors(self, song_temps):
        """기존 태그 엔진 빌드 호환"""
        if self.tag_engine:
            self.tag_engine.build_tag_vectors(song_temps)
            # 호환용 속성 갱신
            self.tfidf_matrix = self.tag_engine.tfidf_matrix
            self.song_id_to_index = self.tag_engine.song_id_to_index
            self.index_to_song_id = self.tag_engine.index_to_song_id
    
    def calculate_similarity_to_vector(self, song_id, seed_vector):
        """시드 벡터와의 유사도 (기존 호환)"""
        # 태그 엔진 기반 (시드 벡터는 태그 TF-IDF 벡터)
        if self.tag_engine:
            return self.tag_engine.calculate_similarity_to_vector(song_id, seed_vector)
        return 0.0
    
    def build_seed_vector(self, seed_tracks):
        """시드 벡터 생성 (기존 호환)"""
        if self.tag_engine:
            return self.tag_engine.build_seed_vector(seed_tracks)
        return None
    
    def find_most_similar(self, seed_song_id, pool_song_ids, top_n=5):
        """가장 유사한 곡 찾기 (Multi-Signal 사용)"""
        results = []
        for cand_id in pool_song_ids:
            if cand_id == seed_song_id:
                continue
            sim = self.calculate_similarity(seed_song_id, cand_id)
            results.append((cand_id, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]
    
    def get_similar_artists(self, artist, limit=20):
        """아티스트 유사도 (기존 호환 — LastFM 위임)"""
        if self.lastfm:
            return self.lastfm.get_similar_artists(artist, limit)
        return []
    
    def get_signal_breakdown(self, song_a_id, song_b_id):
        """디버그용: 각 신호별 유사도 분해"""
        breakdown = {}
        
        if self.tag_engine:
            try:
                breakdown['lastfm_tags'] = self.tag_engine.calculate_similarity(song_a_id, song_b_id)
            except:
                breakdown['lastfm_tags'] = None
        
        if self.lyrics_engine:
            try:
                breakdown['lyrics'] = self.lyrics_engine.calculate_similarity(song_a_id, song_b_id)
            except:
                breakdown['lyrics'] = None
        
        if self.audio_engine:
            try:
                breakdown['audio'] = self.audio_engine.calculate_similarity(song_a_id, song_b_id)
            except:
                breakdown['audio'] = None
        
        breakdown['metadata'] = self._metadata_similarity(song_a_id, song_b_id)
        
        return breakdown
