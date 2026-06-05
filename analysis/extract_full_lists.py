import sys
import pandas as pd
from lifecycle_recommender import run_pipeline
from title_alias import ENGLISH_TO_KOREAN_DISPLAY
import json

def apply_korean_patch(title):
    # 만약 곡 제목 전체가 매핑에 있으면 바로 변경
    if title in ENGLISH_TO_KOREAN_DISPLAY:
        return ENGLISH_TO_KOREAN_DISPLAY[title]
    
    # 일부 괄호나 쓸데없는 태그 제거
    clean_title = title.replace(" - Topic", "").strip()
    if clean_title in ENGLISH_TO_KOREAN_DISPLAY:
        return ENGLISH_TO_KOREAN_DISPLAY[clean_title]
    return title

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

    # 곡 리스트 정리
    rising_songs = []
    cool_songs = []
    rescued_songs = []

    for s in temps.values():
        title_patched = apply_korean_patch(s['title'])
        s['display_title'] = title_patched

        if s['temperature'] == 'Rising':
            rising_songs.append(s)
        elif s['temperature'] == 'Cool':
            cool_songs.append(s)
        
        # 구제받은 노래 판단 로직:
        # 본래 Zone3 (좋다가 질림)이면 Cool로 가야 하나, 아티스트 애정도 등으로 Warm이 된 곡들
        # Zone3 조건: second_half_skip > first_half_skip + 0.15
        s_half_skip = s.get('second_half_skip')
        f_half_skip = s.get('first_half_skip')
        if s_half_skip is not None and f_half_skip is not None:
            is_zone3 = s_half_skip > f_half_skip + 0.15
            if is_zone3 and s['temperature'] == 'Warm':
                rescued_songs.append(s)
                
    # Sort
    rising_songs.sort(key=lambda x: x['total_plays'], reverse=True)
    cool_songs.sort(key=lambda x: x['total_plays'], reverse=True)
    rescued_songs.sort(key=lambda x: x['total_plays'], reverse=True)

    # 출력 포맷터
    def print_section(songs, title, desc=""):
        print(f"\n{'='*90}")
        print(f" {title} ({len(songs)}곡)")
        if desc:
            print(f"   * {desc}")
        print(f"{'='*90}")
        print(f"{'아티스트':<20} | {'곡명':<45} | {'재생':>3} | {'스킵':>3} | {'최근':>3}")
        print("-" * 90)
        for s in songs:
            artist = s['artist'][:20]
            t = s['display_title'][:45]
            plays = s['total_plays']
            skip = int(s['skip_rate'] * 100)
            last = s['days_since_last']
            print(f"{artist:<20} | {t:<45} | {plays:>4} | {skip:>4} | {last:>4}")

    # 리포트 터미널 출력
    print_section(rising_songs, "🔥 Rising (급상승) 전체 리스트", "최근 30일 내에 활발히 듣고 있으며, 들을수록 스킵이 줄어드는 최애곡")
    print_section(cool_songs, "🧊 Cool (식히는 중) 전체 리스트", "예전엔 많이 들었지만 패턴이 질렸음(Zone3)을 보이거나 스킵이 너무 잦은 곡")
    print_section(rescued_songs, "🚑 구제받은 노래 (Zone 3 -> Warm) 리스트", "최근에 질렸음(Zone3)을 보였으나, 아티스트 애정도가 높아 얼리지 않고 순환석에 둔 곡")

    # 파일로도 저장
    with open('최종_분류_리포트.txt', 'w', encoding='utf-8') as f:
        # Redirect stdout to file just for this block
        original_stdout = sys.stdout
        sys.stdout = f
        
        print_section(rising_songs, "🔥 Rising (급상승) 전체 리스트", "최근 30일 내에 활발히 듣고 있으며, 들을수록 스킵이 줄어드는 최애곡")
        print_section(cool_songs, "🧊 Cool (식히는 중) 전체 리스트", "예전엔 많이 들었지만 패턴이 질렸음(Zone3)을 보이거나 스킵이 너무 잦은 곡")
        print_section(rescued_songs, "🚑 구제받은 노래 (Zone 3 -> Warm) 리스트", "최근에 질렸음(Zone3)을 보였으나, 아티스트 애정도가 높아 얼리지 않고 순환석에 둔 곡")
        
        sys.stdout = original_stdout
        
    print("\n✅ 리포트 생성 완료: 최종_분류_리포트.txt에 동일 내용이 저장되었습니다.")

if __name__ == "__main__":
    main()
