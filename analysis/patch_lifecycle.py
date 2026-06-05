import sys
with open(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\lifecycle_recommender.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. _calculate_similarity 수정
import re
new_calc = """    def _calculate_similarity(self, target_song, seed_song=None, seed_vector=None):
        \"\"\"
        [태그 기반 코사인 유사성 로직]
        Last.fm 태그 TF-IDF 벡터의 코사인 유사도(0~1)를 사용하여
        결과를 직관적인 백분율(0~100%)로 반환합니다.
        \"\"\"
        if seed_vector is not None and hasattr(self, 'similarity_engine') and self.similarity_engine:
            target_id = target_song.get('song_id')
            if target_id:
                sim_score = self.similarity_engine.calculate_similarity_to_vector(target_id, seed_vector)
                return float(sim_score) * 100.0

        if not seed_song:
            return 0.0

        if hasattr(self, 'similarity_engine') and self.similarity_engine:
            target_id = target_song.get('song_id')
            seed_id = seed_song.get('song_id')
            if target_id and seed_id:
                sim_score = self.similarity_engine.calculate_similarity(target_id, seed_id)
                return float(sim_score) * 100.0

        # 폴백 로직
        score = 0.0
        target_time = target_song.get('preferred_time')
        seed_time = seed_song.get('preferred_time')
        if target_time and seed_time and target_time == seed_time:
            score += 2.0
        if target_song.get('artist') == seed_song.get('artist'):
            score += 2.0
        elif target_song.get('tier') == seed_song.get('tier'):
            score += 1.0
        target_tod = target_song.get('peak_time_of_day')
        seed_tod = seed_song.get('peak_time_of_day')
        if target_tod and seed_tod and (target_tod == seed_tod) and (target_tod != 'Unknown'):
            score += 2.0
        return score"""

content = re.sub(
    r"    def _calculate_similarity.*?return score",
    new_calc,
    content,
    flags=re.DOTALL
)

# 2. generate_playlist 수정
old_gen = """    def generate_playlist(self, total_songs=20, preset='default', custom_ratios=None):
        \"\"\"
        지정된 비율로 플레이리스트를 생성한다.

        Args:
            total_songs: 생성할 플레이리스트의 곡 수
            preset: 프리셋 이름 ('default', 'explore', 'comfort', 'nostalgia')
            custom_ratios: 사용자 정의 비율 딕셔너리 (preset 대신 사용)
        \"\"\"
        ratios = custom_ratios or self.PRESETS.get(preset, self.PRESETS['default'])

        # 온도별 곡 분류 (Discovery_Candidate는 Discovery 풀에 합류)
        temp_pools = defaultdict(list)
        for song_id, info in self.song_temps.items():
            temp = info['temperature']
            if temp == 'Discovery_Candidate':
                temp_pools['Discovery_Candidate'].append(info)
            else:
                temp_pools[temp].append(info)

        playlist = []

        # 시드(Seed) 곡 선정: Rising에서 가장 많이 들은 곡, 없으면 Steady/Warm 순
        seed_song = None
        seed_candidates = temp_pools.get('Rising', []) + temp_pools.get('Steady', []) + temp_pools.get('Warm', [])
        if seed_candidates:
            seed_song = sorted(seed_candidates, key=lambda x: x['total_plays'], reverse=True)[0]"""

new_gen = """    def generate_playlist(self, total_songs=20, preset='default', custom_ratios=None, seed_tracks=None):
        \"\"\"
        지정된 비율로 플레이리스트를 생성한다.

        Args:
            total_songs: 생성할 플레이리스트의 곡 수
            preset: 프리셋 이름 ('default', 'explore', 'comfort', 'nostalgia')
            custom_ratios: 사용자 정의 비율 딕셔너리 (preset 대신 사용)
            seed_tracks: 시작곡(다수일 경우 리스트 형태 [{'artist':'', 'title':'', 'weight':1.0}])
        \"\"\"
        ratios = custom_ratios or self.PRESETS.get(preset, self.PRESETS['default'])

        temp_pools = defaultdict(list)
        for song_id, info in self.song_temps.items():
            temp = info['temperature']
            if temp == 'Discovery_Candidate':
                temp_pools['Discovery_Candidate'].append(info)
            else:
                temp_pools[temp].append(info)

        playlist = []
        seed_song = None
        seed_vector = None

        if seed_tracks and hasattr(self, 'similarity_engine') and self.similarity_engine:
            seed_vector = self.similarity_engine.build_seed_vector(seed_tracks)
        
        if not seed_vector:
            seed_candidates = temp_pools.get('Rising', []) + temp_pools.get('Steady', []) + temp_pools.get('Warm', [])
            if seed_candidates:
                seed_song = sorted(seed_candidates, key=lambda x: x['total_plays'], reverse=True)[0]"""

content = content.replace(old_gen, new_gen)

# 3. _calculate_similarity 사용하는 곳 수정
content = content.replace("sim_score = self._calculate_similarity(song, seed_song)", "sim_score = self._calculate_similarity(song, seed_song=seed_song, seed_vector=seed_vector)")
content = content.replace("song_copy['similarity_score'] = self._calculate_similarity(song, seed_song)", "song_copy['similarity_score'] = self._calculate_similarity(song, seed_song=seed_song, seed_vector=seed_vector)")
content = content.replace("self._calculate_similarity(song, seed_song), 1)", "self._calculate_similarity(song, seed_song=seed_song, seed_vector=seed_vector), 1)")

# 4. _get_discovery_songs 수정
content = content.replace("def _get_discovery_songs(self, n, seed_song=None):", "def _get_discovery_songs(self, n, seed_song=None, seed_vector=None):")
content = content.replace("discovery_songs = self._get_discovery_songs(n, seed_song=seed_song)", "discovery_songs = self._get_discovery_songs(n, seed_song=seed_song, seed_vector=seed_vector)")

# 5. run_pipeline 수정
content = content.replace("def run_pipeline(csv_path, user_name='사용자', playlist_size=15, preset='default', metadata_path=None, user_birth_year=1998):", "def run_pipeline(csv_path, user_name='사용자', playlist_size=15, preset='default', metadata_path=None, user_birth_year=1998, seed_tracks=None):")
content = content.replace("playlist = mixer.generate_playlist(total_songs=playlist_size, preset=preset)", "playlist = mixer.generate_playlist(total_songs=playlist_size, preset=preset, seed_tracks=seed_tracks)")

with open(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\lifecycle_recommender.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("lifecycle_recommender.py updated!")
