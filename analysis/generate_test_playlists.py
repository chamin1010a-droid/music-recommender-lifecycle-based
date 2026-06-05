import sys
import pandas as pd
from lifecycle_recommender import run_pipeline

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    csv_path = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube 시청 기록\ytm_history_features.csv'
    # 백업 경로 지정
    csv_path = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
    meta_path = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\ytm_metadata_cache.csv'

    print("🎧 사용자 체감 검증용 플레이리스트 추출 🎧\n")

    presets = [
        ('default', '🎵 기본 모드 (밸런스형)'),
        ('explore', '🌍 탐험 모드 (신곡/덜 들은 곡 위주)'),
        ('comfort', '☕ 휴식 모드 (익숙하고 안정적인 곡 위주)')
    ]

    for preset_key, preset_name in presets:
        result = run_pipeline(
            csv_path=csv_path,
            user_name='user',
            playlist_size=15,
            preset=preset_key,
            metadata_path=meta_path,
            user_birth_year=1998
        )
        
        playlist = result.get('playlist', [])
        
        print(f"============================================================")
        print(f"{preset_name}")
        print(f"============================================================")
        
        # 카테고리별 등급 통계 계산
        temp_counts = {}
        for idx, song in enumerate(playlist, 1):
            t = song.get('reason', song.get('temperature', '?'))
            # Discovery의 경우 reason에 긴 문자열이 들어가는 경우 있음
            if 'Discovery' in t:
                t = 'Discovery'
            temp_counts[t] = temp_counts.get(t, 0) + 1
            
            artist = song['artist'].replace(' - Topic', '')
            title = apply_korean_patch(song['title'])
            plays = song['total_plays']
            sim = song.get('similarity_score', 0)
            reason = song.get('reason', song.get('temperature'))
            
            # 이모지 매핑
            emoji = "🔥" if reason in ["Rising", "Steady"] else "🟡" if reason == "Warm" else "🧊" if reason == "Cool" else "❄️" if reason == "Frozen" else "🆕"
            if 'Discovery' in reason:
                emoji = "🆕"
                
            print(f"{idx:>2}. {emoji} {title[:30]:<30} | {artist[:15]:<15}")
            print(f"    └─ [{reason}] (재생: {plays:>2}회) (유사도: {sim}점)")
            
        print("\n--- 구성 비율 ---")
        for k, v in temp_counts.items():
            print(f"  {k}: {v}곡 ({int(v/15*100)}%)")
        print("\n")

if __name__ == '__main__':
    # 패치 함수를 못 부를 경우 대비
    import sys
    import os
    if not os.path.exists('title_alias.py'):
        def apply_korean_patch(t): return t
    else:
        from title_alias import ENGLISH_TO_KOREAN_DISPLAY
        def apply_korean_patch(title):
            if title in ENGLISH_TO_KOREAN_DISPLAY:
                return ENGLISH_TO_KOREAN_DISPLAY[title]
            clean_title = title.replace(" - Topic", "").strip()
            if clean_title in ENGLISH_TO_KOREAN_DISPLAY:
                return ENGLISH_TO_KOREAN_DISPLAY[clean_title]
            return title
            
    main()
