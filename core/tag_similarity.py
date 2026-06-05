import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from lastfm_client import LastFMClient

class TagSimilarityEngine:
    def __init__(self, api_key):
        self.lastfm = LastFMClient(api_key=api_key)
        self.vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b[\w가-힣]+\b')
        self.tfidf_matrix = None
        self.song_id_to_index = {}
        self.index_to_song_id = {}
        
    def build_tag_vectors(self, song_temps: dict):
        """
        전체 곡의 태그를 Last.fm에서 수집하고 TF-IDF 벡터로 변환합니다.
        (처음 실행 시 시간이 꽤 걸릴 수 있으나 API 캐싱으로 인해 2번째부터는 즉시 불러옴)
        """
        corpus = []
        song_ids = []
        
        print("\n[태그 유사도 엔진] Last.fm에서 곡별 특징(태그) 정보 수집 중...")
        extracted_count = 0
        total = len(song_temps)
        
        for song_id, info in song_temps.items():
            artist = info.get('artist', '')
            title = info.get('title', '')
            
            # 태그 수집
            tags = self.lastfm.get_combined_tags(artist, title)
            
            # 텍스트화 (예: "indie rock korean pop")
            if tags:
                tag_doc = " ".join(tags)
            else:
                # 태그를 찾지 못한 경우 아티스트+장르로 폴백
                genre = info.get('genre', '')
                tag_doc = f"{artist.replace(' ','')} {genre}"
                
            corpus.append(tag_doc)
            song_ids.append(song_id)
            
            extracted_count += 1
            if extracted_count % 50 == 0:
                print(f"  진행 상황: {extracted_count} / {total} 곡 처리 완료")
                
        # TF-IDF 학습 및 행렬 변환
        print("  TF-IDF 벡터 피팅 중...")
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        
        # 인덱스 매핑 저장
        for i, sid in enumerate(song_ids):
            self.song_id_to_index[sid] = i
            self.index_to_song_id[i] = sid
            
        print("[태그 유사도 엔진] 준비 완료!\n")
        
    def calculate_similarity(self, song_a_id, song_b_id) -> float:
        """두 곡 간의 코사인 유사도(0.0 ~ 1.0)를 반환"""
        if self.tfidf_matrix is None:
            return 0.0
            
        idx_a = self.song_id_to_index.get(song_a_id)
        idx_b = self.song_id_to_index.get(song_b_id)
        
        if idx_a is None or idx_b is None:
            return 0.0
            
        vec_a = self.tfidf_matrix[idx_a]
        vec_b = self.tfidf_matrix[idx_b]
        
        sim = cosine_similarity(vec_a, vec_b)[0][0]
        return float(sim)

    def find_most_similar(self, seed_song_id, pool_song_ids, top_n=5):
        """특정 곡(seed)과 pool 안에서 가장 유사한 n곡 반환"""
        if self.tfidf_matrix is None or seed_song_id not in self.song_id_to_index:
            return []
            
        idx_seed = self.song_id_to_index[seed_song_id]
        vec_seed = self.tfidf_matrix[idx_seed]
        
        # 전체 곡에 대한 유사도 배열 계산
        similarities = cosine_similarity(vec_seed, self.tfidf_matrix)[0]
        
        results = []
        for cand_id in pool_song_ids:
            if cand_id == seed_song_id:
                continue
            cand_idx = self.song_id_to_index.get(cand_id)
            if cand_idx is not None:
                sim = float(similarities[cand_idx])
                results.append((cand_id, sim))
                
        # 유사도 내림차순 정렬
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

    def build_seed_vector(self, seed_tracks: list):
        """
        여러 시작곡의 태그들을 수집해 가상 평균 TF-IDF 벡터를 생성합니다.
        seed_tracks는 딕셔너리 리스트: [{'artist': '...', 'title': '...', 'weight': 1.0}, ...]
        """
        if not self.vectorizer or not hasattr(self.vectorizer, 'vocabulary_'):
            return None
            
        vectors = []
        weights = []
        
        for track in seed_tracks:
            artist = track.get('artist', '')
            title = track.get('title', '')
            weight = track.get('weight', 1.0)
            
            # 태그 수집
            tags = self.lastfm.get_combined_tags(artist, title)
            if tags:
                tag_doc = " ".join(tags)
            else:
                tag_doc = f"{artist.replace(' ','')}"
                
            vec = self.vectorizer.transform([tag_doc])
            vectors.append(vec.toarray()[0])
            weights.append(weight)
            
        if not vectors:
            return None
            
        # 가중 평균 계산
        vectors = np.array(vectors)
        weights = np.array(weights).reshape(-1, 1)
        
        weighted_sum = np.sum(vectors * weights, axis=0)
        total_weight = np.sum(weights)
        
        avg_vector = weighted_sum / total_weight if total_weight > 0 else weighted_sum
        
        from scipy.sparse import csr_matrix
        return csr_matrix([avg_vector])

    def calculate_similarity_to_vector(self, song_id, seed_vector) -> float:
        """특정 곡(song_id)과 주어진 가상 평균 벡터 간의 유사도 반환"""
        if self.tfidf_matrix is None or seed_vector is None:
            return 0.0
            
        idx = self.song_id_to_index.get(song_id)
        if idx is None:
            return 0.0
            
        vec_target = self.tfidf_matrix[idx]
        sim = cosine_similarity(vec_target, seed_vector)[0][0]
        return float(sim)
