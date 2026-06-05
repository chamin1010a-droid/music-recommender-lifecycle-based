import pandas as pd
import numpy as np
import codecs, sys
import os

sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

df = pd.read_csv(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

target_keywords = [
    "클라우드 쿠쿠 랜드",
    "the volunteers",
    "wont go home without you",
    "결혼까지 생각했어",
    "그땐 그땐 그땐"
]

print("=== 특정 곡 심층 분석 리포트 ===\n")

for keyword in target_keywords:
    # 키워드가 포함된 곡 검색 (대소문자 무시)
    mask = df['song_id'].str.contains(keyword, case=False, na=False)
    matches = df[mask]
    
    if matches.empty:
        print(f"[{keyword}] 에 해당하는 데이터를 찾을 수 없습니다.\n")
        continue
    
    # 여러 버전이 있을 수 있으므로 가장 많이 재생된 song_id 선택
    best_match_song_id = matches['song_id'].value_counts().index[0]
    song_df = df[df['song_id'] == best_match_song_id].sort_values('timestamp').reset_index(drop=True)
    
    total_plays = len(song_df)
    skips = song_df['is_skipped'].sum()
    skip_rate = (skips / total_plays) * 100
    
    # 전반부 vs 후반부 스킵 변화
    half = total_plays // 2
    first_half_skip = song_df.iloc[:half]['is_skipped'].mean() if half > 0 else 0
    second_half_skip = song_df.iloc[half:]['is_skipped'].mean() if half > 0 else song_df['is_skipped'].mean()

    print(f"■ 곡명: {best_match_song_id}")
    print(f"  - 총 재생: {total_plays}회")
    print(f"  - 전체 스킵률: {skip_rate:.1f}% ({skips}회 스킵)")
    
    # 패턴 진단
    if total_plays >= 5:
        if first_half_skip > second_half_skip + 0.2:
            diag = "🟡 듣다 보니 좋아진 '골디락스' 패턴 (초반 거부감 -> 후반 적응)"
        elif second_half_skip > first_half_skip + 0.2:
            diag = "🔴 처음엔 좋았으나 최근 지루해진 '피로도' 패턴"
        elif skip_rate < 20:
            diag = "🟢 처음부터 지금까지 쭉 믿고 듣는 '안정적 취향' 패턴"
        else:
            diag = "⚪ 큰 변화 없이 무난하게 듣는 '습관적 청취' 패턴"
        print(f"  - 분석 결과: {diag}")
    else:
        print("  - 분석 결과: 데이터 부족 (재생 5회 미만)")
    
    # 타임라인 요약
    first_play = song_df.iloc[0]['timestamp'].strftime('%Y-%m-%d')
    last_play = song_df.iloc[-1]['timestamp'].strftime('%Y-%m-%d')
    print(f"  - 감상 기간: {first_play} ~ {last_play}")
    print()
