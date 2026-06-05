"""
음악 추천 알고리즘 — 통합 실행 진입점
=====================================
이 스크립트 하나로 전체 파이프라인이 동작합니다.

사용법:
  python run.py              # 기존 모드 (수동 호감도)
  python run.py lgbm         # LightGBM 모드 (학습된 호감도)
  python run.py explore      # 탐색 모드
  python run.py comfort      # 안정 모드

파이프라인 흐름:
  Step 0: 아티스트명 정규화 (Layer 0)
  Step 1: 아티스트 Tier 분류 (Layer 1: K-Means 클러스터링)
  Step 2: 곡별 온도 판별 (Layer 2: Zone 기반 + 아티스트 생존율 보정)
  Step 3: 비대칭 흐름 감지 (성장 Bottom-Up / 쇠퇴 Top-Down)
  Step 4: 태그 유사도 엔진 (Last.fm TF-IDF)
  Step 5: 외부 신곡 발굴 엔진 (Last.fm 유사 아티스트 + 3단계 Discovery)
  Step 6: 플레이리스트 생성 (내부 + 외부 Discovery 통합 믹싱)
"""

import os
import sys

# 새 폴더 구조에 맞게 Python Path 추가
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'core'))

from lifecycle_recommender import run_pipeline


def main():
    """user 본인 데이터로 통합 파이프라인 실행"""
    
    # 프리셋 선택 (커맨드라인 인자)
    preset = 'default'
    if len(sys.argv) > 1:
        preset = sys.argv[1]
        if preset not in ('default', 'explore', 'comfort', 'lgbm'):
            print(f"⚠️ 알 수 없는 프리셋: {preset}")
            print(f"  사용 가능: default, explore, comfort, lgbm")
            return
    
    # 데이터 경로 (업데이트된 features CSV)
    target_csv = os.path.join(
        BASE_DIR, '유튜브 뮤직 로그들', 'user', 'user_features.csv'
    )
    metadata_path = os.path.join(BASE_DIR, 'data', 'caches', 'ytm_metadata_cache.csv')
    
    if not os.path.exists(target_csv):
        print(f"❌ 분석 대상 파일이 없습니다: {target_csv}")
        return
    
    if not os.path.exists(metadata_path):
        metadata_path = None
    
    print(f"🎧 프리셋: {preset}")
    
    # 통합 파이프라인 실행 (내부 + 외부 Discovery 자동 통합)
    result = run_pipeline(
        csv_path=target_csv,
        user_name='user',
        playlist_size=20,
        preset=preset,
        metadata_path=metadata_path,
        user_birth_year=1998
    )
    
    # 결과 요약
    playlist = result['playlist']
    internal_count = sum(1 for s in playlist if s.get('discovery_source') == 'internal')
    external_count = sum(1 for s in playlist if s.get('discovery_source') == 'external')
    known_count = len(playlist) - internal_count - external_count
    
    print(f"\n{'='*60}")
    print(f"📋 최종 요약 (프리셋: {preset})")
    print(f"{'='*60}")
    print(f"  총 {len(playlist)}곡 추천")
    print(f"  - 기존 곡 (Rising/Steady/Warm): {known_count}곡")
    print(f"  - 내부 발견 (라이브러리 재발굴): {internal_count}곡")
    print(f"  - 외부 신곡 (Last.fm 신규 발굴): {external_count}곡 🆕")
    

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
