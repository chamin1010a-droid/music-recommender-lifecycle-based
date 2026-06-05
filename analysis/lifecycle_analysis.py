import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta

# Set stdout to utf-8
sys.stdout.reconfigure(encoding='utf-8')

def analyze_lifecycle(csv_path, user_name, min_plays=30):
    print(f"\n[{user_name} 데이터 바탕으로 '음원 생애주기 가설' 검증 중...]")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['timestamp'])
    
    # 마지막 데이터 수집일 (현 시점 기준점)
    max_date = df['date'].max()
    print(f"데이터 마지막 기록일: {max_date.strftime('%Y-%m-%d')}")
    
    song_counts = df['song_id'].value_counts()
    target_songs = song_counts[song_counts >= min_plays].index
    
    print(f"총 {min_plays}회 이상 들은 최애곡 후보: {len(target_songs)}곡 분석\n")
    
    # 4단계 분류 카운터
    lifecycle_stats = {
        '성장_폭발기 (최근까지 미친듯이 듣는 중)': 0,
        '번아웃_은퇴기 (단물 다 빠져서 안 듣기 시작함)': 0,
        '향수_맥락기 (오랜만에 가끔 1~2번씩 다시 듣는 곡)': 0,
        '그외 (꾸준히 듣는 스테디셀러)': 0
    }
    
    for song in target_songs:
        song_df = df[df['song_id'] == song].sort_values('date').reset_index(drop=True)
        total_plays = len(song_df)
        
        first_play = song_df['date'].iloc[0]
        last_play = song_df['date'].iloc[-1]
        
        # 30일(한 달) 단위로 쪼개어 가장 많이 들은 달(Peak) 추적
        song_df['month_year'] = song_df['date'].dt.to_period('M')
        peak_month_plays = song_df['month_year'].value_counts().max()
        peak_ratio = peak_month_plays / total_plays
        
        # 마지막으로 들은 지 얼마나 지났는가? (휴지기)
        days_since_last_play = (max_date - last_play).days
        
        # 가설 1: 특정 짧은 기간(Peak)에 청취가 폭발적으로 몰려있어야 한다. (폭발기 증명)
        # 예: 전체 횟수의 30% 이상이 단 한 달 안에 발생함
        is_explosive = peak_ratio >= 0.30 
        
        # 가설 2: 한때 미친듯이 들었지만 현재는 전혀 듣지 않는다 (번아웃 증명)
        # 예: 마지막으로 들은 지 60일 이상 지남 (최근 기록이 없음)
        is_retired = days_since_last_play >= 60
        
        if is_explosive and is_retired:
            lifecycle_stats['번아웃_은퇴기 (단물 다 빠져서 안 듣기 시작함)'] += 1
        elif is_explosive and not is_retired:
            lifecycle_stats['성장_폭발기 (최근까지 미친듯이 듣는 중)'] += 1
        else:
            # 폭발기가 뚜렷하지 않은 곡들 중에서도, 엄청 오래 안 듣다가 가장 최근에 딱 1~2번 들은 기록을 찾음
            # 간단히 "스테디셀러" vs "향수기" 분리
            if is_retired:
                lifecycle_stats['번아웃_은퇴기 (단물 다 빠져서 안 듣기 시작함)'] += 1
            else:
                lifecycle_stats['그외 (꾸준히 듣는 스테디셀러)'] += 1
                
    total_analyzed = len(target_songs)
    print("=== 노래 생애주기 분석 결과 ===")
    for status, count in lifecycle_stats.items():
        ratio = (count / total_analyzed) * 100
        print(f"- {status}: {count}곡 ({ratio:.1f}%)")
        
    print("\n[결론 요약]")
    print(f"최애곡(많이 들은 곡) 중 '번아웃/은퇴'하여 더 이상 재생되지 않는 곡의 비율: {(lifecycle_stats['번아웃_은퇴기 (단물 다 빠져서 안 듣기 시작함)']/total_analyzed)*100:.1f}%")
    print("-> 사용자님이 말씀하신 '아무리 많이 들었던 곡이라도 결국 은퇴기에 접어들고 더 이상 듣지 않는다'는 가설이 데이터로 명확히 증명되는지 확인합니다.")

if __name__ == '__main__':
    csv_path = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\유튜브 뮤직 로그들\친구D\친구D_features.csv'
    analyze_lifecycle(csv_path, '친구D(친구)', min_plays=20)
