import pandas as pd
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

csv_path = r"c:\Users\user\Desktop\데이터분석\음악 프로젝트\유튜브 뮤직 로그들\친구A\친구A_features.csv"
df = pd.read_csv(csv_path, encoding='utf-8-sig')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

print("--- [친구A] 1곡 반복재생 미기록 가설 검증 ---\n")

# 사용자가 가장 많이 들은 곡들 추출
top_songs = df['song_id'].value_counts().head(5).index

print("가설: '한 곡 반복'을 켜두면 기록 상 한 번만 남고, 대신 [다음 곡으로 넘어갈 때까지의 시간 격차(Time Gap)]가 매우 길어질 것이다.\n")

for song in top_songs:
    s_df = df[df['song_id'] == song].copy()
    
    # 1시간(3600초) 이상 차이나는 경우는 그냥 음악을 끄고 나중에 다시 킨 경우(세션 종료)로 간주
    # 한 곡을 반복재생하고 이어서 다른 노래를 들었다면, 격차가 10분~50분 사이일 확률이 높음.
    
    print(f"[{song[:40]}] (총 재생: {len(s_df)}회)")
    
    # 다음 곡으로 넘어간 시간(time_gap)의 분포 확인
    gaps = s_df['time_gap'].dropna() / 60  # 분 단위 변환
    
    # 곡 재생 후 1분~10분 사이 (딱 한 번 정상 재생하고 다음 곡으로 넘어간 경우)
    normal_plays = gaps[(gaps >= 1) & (gaps <= 10)]
    
    # 곡 재생 후 10분~60분 사이 (반복 재생 의심 구간: 노래 1곡이 10분이 넘진 않으므로)
    loop_suspects = gaps[(gaps > 10) & (gaps <= 60)]
    
    print(f"  - 한 번만 듣고 바로 다음 곡으로 넘어간 횟수(1~10분 갭): {len(normal_plays)}회")
    print(f"  - 반복 재생 의심 횟수(10~60분 갭): {len(loop_suspects)}회")
    
    if len(loop_suspects) > 0:
        print(f"  * 의심되는 갭 타임(분): {list(np.round(loop_suspects.values, 1))}")
    print()
    
print(">> YouTube Music 데이터 로깅 특성:")
print("유튜브 뮤직은 시스템 구조상 '한 곡 반복(Loop 1)'을 설정해도 매 재생 사이클이 시작될 때마다 스트리밍 카운트를 위해 새로운 시청 기록(timestamp)을 남깁니다.")
print("다만, 오프라인 저장본을 비행기 모드 등 데이터가 완전히 끊긴 상태에서 무한 반복한 경우에는 기록이 누락될 수 있습니다.")
