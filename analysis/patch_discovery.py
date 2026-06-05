import sys
with open(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\external_discovery.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. discover_new_songs 수정
old_def = "def discover_new_songs(self, n=15, min_similarity=0.25, max_similarity=0.85,\n                           discovery_preset='default'):"
new_def = "def discover_new_songs(self, n=15, min_similarity=0.25, max_similarity=0.85,\n                           discovery_preset='default',\n                           seed_tracks=None, seed_vector=None, similarity_engine=None):"
content = content.replace(old_def, new_def)

# 2. 유사 아티스트 탐색 부분 수정
old_top_artists = """        # 1단계: 유사 아티스트 탐색
        top_artists = self._get_top_artists(n=15)
        print(f"  내 TOP 아티스트 {len(top_artists)}명에서 유사 가수 탐색 중...")"""

new_top_artists = """        # 1단계: 유사 아티스트 탐색
        if seed_tracks:
            top_artists = list(set([t.get('artist', '').replace(' - Topic', '').strip() for t in seed_tracks if t.get('artist')]))
            print(f"  시작곡 아티스트({len(top_artists)}명)에서 유사 가수 탐색 중...")
        else:
            top_artists = self._get_top_artists(n=15)
            print(f"  내 TOP 아티스트 {len(top_artists)}명에서 유사 가수 탐색 중...")"""
content = content.replace(old_top_artists, new_top_artists)

# 3. _calculate_tag_similarity 교체 (seed_vector가 있으면 그것과 비교)
old_sim = """    def _calculate_tag_similarity(self, candidate_tags, my_tag_docs):
        \"\"\"
        후보 곡의 태그와 내 TOP 50곡 태그들의 TF-IDF 코사인 유사도 평균
        \"\"\"
        if not candidate_tags:
            return 0.0
        if not my_tag_docs:
            return 0.0
        
        all_docs = my_tag_docs + [" ".join(candidate_tags)]
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(all_docs)
        
        candidate_vec = tfidf[-1]
        library_vecs = tfidf[:-1]
        
        similarities = cosine_similarity(candidate_vec, library_vecs)[0]
        return float(np.mean(similarities))"""

new_sim = """    def _calculate_tag_similarity(self, candidate_tags, my_tag_docs, seed_vector=None, similarity_engine=None):
        \"\"\"
        후보 곡의 태그와 기준 태그 벡터와의 유사도를 계산.
        seed_vector가 주어지면 그것과 직접 비교, 없으면 내 라이브러리와 비교
        \"\"\"
        if not candidate_tags:
            return 0.0
            
        if seed_vector is not None and similarity_engine is not None and similarity_engine.vectorizer is not None:
            # Seed Vector와 직접 비교
            tag_doc = " ".join(candidate_tags)
            cand_vec = similarity_engine.vectorizer.transform([tag_doc])
            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity(cand_vec, seed_vector)[0][0]
            return float(sim)
            
        if not my_tag_docs:
            return 0.0
        
        all_docs = my_tag_docs + [" ".join(candidate_tags)]
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(all_docs)
        
        candidate_vec = tfidf[-1]
        library_vecs = tfidf[:-1]
        
        similarities = cosine_similarity(candidate_vec, library_vecs)[0]
        return float(np.mean(similarities))"""

content = content.replace(old_sim, new_sim)

# 4. _calculate_tag_similarity 호출 부분 수정
content = content.replace("similarity = self._calculate_tag_similarity(track_tags, my_tag_docs)", "similarity = self._calculate_tag_similarity(track_tags, my_tag_docs, seed_vector, similarity_engine)")

with open(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\external_discovery.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("external_discovery.py updated!")
