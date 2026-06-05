import sys
import pandas as pd
from lifecycle_recommender import run_pipeline
from title_alias import ENGLISH_TO_KOREAN_DISPLAY

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    csv_path = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
    meta_path = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\ytm_metadata_cache.csv'

    result = run_pipeline(
        csv_path=csv_path,
        user_name='user',
        playlist_size=15,
        preset='default',
        metadata_path=meta_path,
        user_birth_year=1998
    )
    temps = result['temp_tracker'].song_temps
    artist_survival = result['temp_tracker'].artist_survival

    print('=== Cool 등급 판정 사유 세부 분석 ===\n')

    reasons = {'Zone3 (최근 질림)': 0, 'Skip Rate > 0.5 (스킵 과다)': 0, '가속 강등 (아티스트 폼 하락)': 0, '기타 (Zone4, 재생적음 등)': 0}
    
    for s in temps.values():
        if s['temperature'] == 'Cool':
            f_skip = s.get('first_half_skip')
            s_skip = s.get('second_half_skip')
            skip_total = s['skip_rate']
            surv = artist_survival.get(s['artist'], 0)
            
            # 판정 추적 로직 (비슷한 순서로)
            reason = ''
            
            # 가속 강등 (기본적으로 Warm이었어야 했는데 surv < 0.15 로 강등된 경우)
            # 엄밀히는 역추적하기 어려우나, surv < 0.15 이고 Zone3가 아닌 경우일 확률이 높음.
            if surv < 0.15 and not (s_skip is not None and f_skip is not None and s_skip > f_skip + 0.15):
                reason = '가속 강등 (아티스트 폼 하락)'
            elif s_skip is not None and f_skip is not None and s_skip > f_skip + 0.15:
                # Zone 3 조건
                if s['days_since_last'] < 90:
                    reason = 'Zone3 (최근 질림)'
                else: 
                    # 90일 넘었으면 Frozen인데 왜 Cool인지 (가속강등이거나 폴백)
                    reason = '기타 (Zone4, 재생적음 등)'
            elif skip_total > 0.5:
                reason = 'Skip Rate > 0.5 (스킵 과다)'
            else:
                reason = '기타 (Zone4, 재생적음 등)'
                
            reasons[reason] += 1
            
            # 20회 이상 재생된 주요 곡 출력
            if s['total_plays'] >= 20:
                artist = s['artist'][:18].replace(' - Topic', '')
                title = ENGLISH_TO_KOREAN_DISPLAY.get(s['title'], s['title'])[:30]
                print(f"[{reason[:10]}] {artist:<18} | {title:<30} (재생: {s['total_plays']}) (스킵: {int(skip_total*100)}%) (생존율: {int(surv*100)}%)")

    print('\n=== 사유별 총 통계 (76곡 중) ===')
    for k, v in reasons.items():
        print(f'{k}: {v}곡')

if __name__ == '__main__':
    main()
