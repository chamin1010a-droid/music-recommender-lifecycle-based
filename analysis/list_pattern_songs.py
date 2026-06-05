import pandas as pd
import numpy as np
import codecs, sys
import os

sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

df = pd.read_csv(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

song_counts = df['song_id'].value_counts()
songs_over_20 = song_counts[song_counts >= 20].index

patterns = {
    'loved_from_start': [],
    'grew_on_me': [],
    'always_meh': [],
    'loved_then_tired': []
}

for song in songs_over_20:
    song_df = df[df['song_id'] == song].sort_values('timestamp').reset_index(drop=True)
    total_plays = len(song_df)
    
    half = total_plays // 2
    first_half_skip = song_df.iloc[:half]['is_skipped'].mean()
    second_half_skip = song_df.iloc[half:]['is_skipped'].mean()
    
    first_3_skip = song_df.iloc[:min(3, total_plays)]['is_skipped'].mean()
    overall_skip = song_df['is_skipped'].mean()
    
    song_info = f"{song} (총 {total_plays}회 재생)"
    
    if first_3_skip <= 0.33 and overall_skip <= 0.3:
        patterns['loved_from_start'].append(song_info)
    elif first_half_skip > second_half_skip + 0.15:
        patterns['grew_on_me'].append(f"{song_info} [스킵률: {first_half_skip:.0%} -> {second_half_skip:.0%}]")
    elif second_half_skip > first_half_skip + 0.15:
        patterns['loved_then_tired'].append(f"{song_info} [스킵률: {first_half_skip:.0%} -> {second_half_skip:.0%}]")
    else:
        patterns['always_meh'].append(f"{song_info} [평균 스킵률: {overall_skip:.0%}]")

# 결과 출력
print("# 음악 청취 패턴별 곡 리스트 (재생 20회 이상 곡 대상)\n")

print("## 🟢 1. 처음부터 좋았던 곡 (57.4%, 183곡)")
print("> 첫 3회 이내에 거의 스킵하지 않았고, 전체적으로도 스킵이 적은 '확신의 취향' 곡들입니다.\n")
for s in sorted(patterns['loved_from_start']):
    print(f"- {s}")

print("\n## 🟡 2. 듣다 보니 좋아진 곡 (10.7%, 34곡)")
print("> 초반에는 자주 넘겼으나, 노출이 반복될수록 스킵이 줄어든 '골디락스' 스타일 곡들입니다.\n")
for s in sorted(patterns['grew_on_me']):
    print(f"- {s}")

print("\n## 🔴 3. 좋았다가 질린 곡 (6.6%, 21곡)")
print("> 초반에는 잘 듣다가 나중에 스킵이 갑자기 늘어난, '유통기한이 다한' 곡들입니다.\n")
for s in sorted(patterns['loved_then_tired']):
    print(f"- {s}")

print("\n## ⚪ 4. 그냥저냥인 곡 (25.4%, 81곡)")
print("> 스킵률에 큰 변화가 없거나 꾸준히 중간 정도의 빈도로 넘기며 듣는 곡들입니다.\n")
for s in sorted(patterns['always_meh']):
    print(f"- {s}")
