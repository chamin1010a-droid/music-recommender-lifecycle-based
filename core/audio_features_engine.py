"""
[오디오 특성 기반 유사도 엔진]
yt-dlp로 YouTube 음원을 다운받고, librosa로 음향 특성을 추출하여
곡 간 코사인 유사도를 계산합니다.

추출 특성 (37차원):
  - MFCC (20차원): 음색/질감
  - Chroma (12차원): 화성/코드 진행
  - Spectral Centroid (1차원): 밝기
  - Spectral Rolloff (1차원): 고주파 비율
  - Spectral Contrast mean (1차원): 주파수 대비
  - Tempo/BPM (1차원): 빈박수
  - RMS Energy (1차원): 음량/에너지
"""

import os
import json
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

class AudioFeaturesEngine:
    def __init__(self, cache_dir=None, features_cache_file='audio_features_cache.json'):
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), '..', 'data', 'audio_cache'
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.features_cache_file = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'caches', features_cache_file
        )
        os.makedirs(os.path.dirname(self.features_cache_file), exist_ok=True)
        
        self.features = {}  # song_id → numpy array (37-dim)
        self._feature_matrix = None  # 전체 행렬 (N × 37)
        self._song_id_to_idx = {}
        self._load_cache()
    
    def _load_cache(self):
        """캐시에서 이전에 추출한 특성 로드"""
        if os.path.exists(self.features_cache_file):
            try:
                with open(self.features_cache_file, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                self.features = {k: np.array(v) for k, v in raw.items()}
                print(f"  [오디오 엔진] 캐시 로드: {len(self.features)}곡")
            except Exception as e:
                print(f"  [오디오 엔진] 캐시 로드 실패: {e}")
                self.features = {}
    
    def _save_cache(self):
        """추출한 특성을 JSON으로 저장"""
        raw = {k: v.tolist() for k, v in self.features.items()}
        with open(self.features_cache_file, 'w', encoding='utf-8') as f:
            json.dump(raw, f)
    
    def _safe_filename(self, song_id):
        """song_id를 안전한 파일명으로 변환 (해시 사용)"""
        import hashlib
        return hashlib.md5(song_id.encode('utf-8')).hexdigest()
    
    def download_audio(self, song_id, title, artist, max_duration=30):
        """
        YouTube Music에서 제목+아티스트로 검색하여 오디오를 다운로드.
        Returns: 파일 경로 또는 None
        """
        safe_name = self._safe_filename(song_id)
        output_path = os.path.join(self.cache_dir, f"{safe_name}.wav")
        
        # 이미 다운로드된 경우 스킵
        if os.path.exists(output_path):
            return output_path
        
        # 다른 확장자로 존재하는지 확인
        for ext in ['.opus', '.webm', '.m4a', '.mp3']:
            if os.path.exists(os.path.join(self.cache_dir, f"{safe_name}{ext}")):
                return os.path.join(self.cache_dir, f"{safe_name}{ext}")
        
        try:
            import yt_dlp
            
            # 아티스트 정제
            clean_artist = str(artist).replace(' - Topic', '').strip()
            clean_title = str(title).strip()
            
            # YouTube Music에서 검색
            search_query = f"ytsearch1:{clean_title} {clean_artist} official audio"
            temp_path = os.path.join(self.cache_dir, f"{safe_name}_temp")
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': temp_path + '.%(ext)s',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '128',
                }],
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'noplaylist': True,
                'match_filter': yt_dlp.utils.match_filter_func("duration < 600"),
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([search_query])
            
            # wav 파일 찾기
            wav_file = temp_path + '.wav'
            if os.path.exists(wav_file):
                os.rename(wav_file, output_path)
                return output_path
            
            # 다른 확장자로 생성된 경우 (librosa가 읽을 수 있음)
            for ext in ['.opus', '.webm', '.m4a', '.mp3']:
                candidate = temp_path + ext
                if os.path.exists(candidate):
                    final = os.path.join(self.cache_dir, f"{safe_name}{ext}")
                    os.rename(candidate, final)
                    return final
            
            return None
            
        except Exception as e:
            return None
    
    def extract_features(self, audio_path):
        """
        librosa로 오디오 특성 추출.
        Returns: 37차원 numpy array 또는 None
        """
        try:
            import librosa
            
            # 오디오 로드 (모노, 22050Hz, 최대 30초)
            y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=30)
            
            if len(y) < sr * 2:  # 2초 미만이면 무시
                return None
            
            # 1. MFCC (20차원) — 음색/질감
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_mean = np.mean(mfcc, axis=1)  # (20,)
            
            # 2. Chroma (12차원) — 화성
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)  # (12,)
            
            # 3. Spectral Centroid — 밝기
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            centroid_mean = np.mean(centroid)
            
            # 4. Spectral Rolloff — 고주파 비율
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            rolloff_mean = np.mean(rolloff)
            
            # 5. Spectral Contrast — 주파수 대비
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            contrast_mean = np.mean(contrast)
            
            # 6. Tempo/BPM — 빈박수
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo[0])
            
            # 7. RMS Energy — 음량
            rms = librosa.feature.rms(y=y)
            rms_mean = np.mean(rms)
            
            # 전체 벡터 구성 (37차원)
            feature_vec = np.concatenate([
                mfcc_mean,           # 20
                chroma_mean,         # 12
                [centroid_mean],     # 1
                [rolloff_mean],      # 1
                [contrast_mean],     # 1
                [tempo],             # 1
                [rms_mean],          # 1
            ])                       # = 37
            
            return feature_vec
            
        except Exception as e:
            return None
    
    def build_features(self, song_dict, batch_save_interval=50):
        """
        전체 곡에 대해 오디오 특성 추출 (캐시에 없는 곡만).
        song_dict: {song_id: {title, artist, ...}}
        """
        to_process = [
            sid for sid in song_dict.keys() 
            if sid not in self.features
        ]
        
        if not to_process:
            print(f"  [오디오 엔진] 모든 곡이 캐시에 있음 ({len(self.features)}곡)")
            self._build_matrix()
            return
        
        print(f"\n[오디오 특성 엔진] {len(to_process)}곡 처리 시작...")
        print(f"  (이미 캐시: {len(self.features)}곡, 남은 작업: {len(to_process)}곡)")
        
        success = 0
        fail = 0
        start_time = time.time()
        
        for i, song_id in enumerate(to_process):
            info = song_dict.get(song_id, {})
            title = info.get('title', song_id.split(' - ')[0])
            artist = info.get('artist', '')
            
            # 1. YouTube 검색 + 다운로드
            audio_path = self.download_audio(song_id, title, artist)
            
            if audio_path and os.path.exists(audio_path):
                # 2. 특성 추출
                features = self.extract_features(audio_path)
                
                if features is not None:
                    self.features[song_id] = features
                    success += 1
                else:
                    fail += 1
                
                # 다운로드한 파일 삭제 (디스크 절약)
                try:
                    os.remove(audio_path)
                except:
                    pass
            else:
                fail += 1
            
            # 진행 상황
            if (i + 1) % batch_save_interval == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(to_process) - i - 1) / rate if rate > 0 else 0
                print(f"  진행: {i+1}/{len(to_process)} "
                      f"(성공 {success}, 실패 {fail}) "
                      f"| 남은 시간: {remaining/60:.0f}분")
                self._save_cache()
        
        self._save_cache()
        self._build_matrix()
        
        elapsed = time.time() - start_time
        print(f"\n[오디오 엔진] 완료! 성공 {success}곡, 실패 {fail}곡 "
              f"(소요: {elapsed/60:.1f}분)")
    
    def _build_matrix(self):
        """전체 특성을 행렬로 구성 (빠른 유사도 계산용)"""
        if not self.features:
            return
        
        from sklearn.preprocessing import StandardScaler
        
        song_ids = list(self.features.keys())
        matrix = np.array([self.features[sid] for sid in song_ids])
        
        # StandardScaler(z-score): 음수 포함 → 코사인 유사도 차별력 확보
        # MinMaxScaler(0~1)는 모든 값이 양수라 유사도가 0.93~0.98에 몰림
        scaler = StandardScaler()
        self._feature_matrix = scaler.fit_transform(matrix)
        self._scaler = scaler
        
        self._song_id_to_idx = {sid: i for i, sid in enumerate(song_ids)}
    
    def calculate_similarity(self, song_a_id, song_b_id):
        """두 곡 간의 오디오 특성 코사인 유사도 (0~1)"""
        if self._feature_matrix is None:
            return None
        
        idx_a = self._song_id_to_idx.get(song_a_id)
        idx_b = self._song_id_to_idx.get(song_b_id)
        
        if idx_a is None or idx_b is None:
            return None
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        vec_a = self._feature_matrix[idx_a].reshape(1, -1)
        vec_b = self._feature_matrix[idx_b].reshape(1, -1)
        
        sim = cosine_similarity(vec_a, vec_b)[0][0]
        return float(max(0, sim))  # 음수 방지
    
    def calculate_similarity_to_vector(self, song_id, target_vector):
        """곡과 주어진 벡터 간의 유사도"""
        if self._feature_matrix is None:
            return None
        
        idx = self._song_id_to_idx.get(song_id)
        if idx is None:
            return None
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        vec = self._feature_matrix[idx].reshape(1, -1)
        sim = cosine_similarity(vec, target_vector.reshape(1, -1))[0][0]
        return float(max(0, sim))


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    # 테스트: 몇 곡만 처리
    import pandas as pd
    csv_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
    df = pd.read_csv(csv_p)
    
    # 상위 5곡 테스트
    top5 = df.groupby('song_id').size().sort_values(ascending=False).head(5)
    print(f"테스트 곡: {top5.index.tolist()}")
    
    engine = AudioFeaturesEngine()
    
    test_dict = {}
    for sid in top5.index:
        row = df[df['song_id'] == sid].iloc[0]
        test_dict[sid] = {'title': row['title'], 'artist': row['artist']}
    
    engine.build_features(test_dict)
    
    # 유사도 테스트
    ids = list(test_dict.keys())
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            sim = engine.calculate_similarity(ids[i], ids[j])
            if sim is not None:
                a_title = test_dict[ids[i]]['title'][:20]
                b_title = test_dict[ids[j]]['title'][:20]
                print(f"  {a_title} ↔ {b_title}: {sim:.3f}")
