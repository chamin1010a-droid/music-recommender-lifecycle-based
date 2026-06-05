import pandas as pd
import numpy as np
from datetime import datetime
import os
import re
from dateutil import parser

folder_path = r"c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록"
input_csv = os.path.join(folder_path, "ytm_history_parsed.csv")
output_csv = os.path.join(folder_path, "ytm_history_features.csv")

print("Loading data...")
df = pd.read_csv(input_csv)

def parse_korean_date_mixed(time_str):
    try:
        time_str = str(time_str).replace('KST', '').strip().replace('AM', '오전').replace('PM', '오후')
        match = re.search(r'(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.\s*(오전|오후)\s*(\d{1,2}):(\d{2}):(\d{2})', time_str)
        if match:
            y, m, d, ampm, h, min_, sec = match.groups()
            h = int(h)
            if ampm == '오후' and h < 12: h += 12
            elif ampm == '오전' and h == 12: h = 0
            return datetime(int(y), int(m), int(d), h, int(min_), int(sec))
    except:
        pass
    try:
        return parser.parse(str(time_str), fuzzy=True)
    except:
        return pd.NaT

print("Parsing timestamps...")
df['timestamp'] = df['time_str'].apply(parse_korean_date_mixed)
df = df.dropna(subset=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
df['song_id'] = df['title'].astype(str) + " - " + df['artist'].fillna("Unknown")

# ================================================
# 1. Time Gap 계산
# ================================================
df['next_timestamp'] = df['timestamp'].shift(-1)
df['time_gap_seconds'] = (df['next_timestamp'] - df['timestamp']).dt.total_seconds()

# ================================================
# 2. 스킵 타이밍 3단계 분류 (고도화)
#
# skip_type:
#   0 = "완주"       : 30초 초과 → 정상 감상 (유효 플레이)
#   1 = "샘플 스킵"  : 10초 초과 ~ 30초 이하 → 들어봤지만 넘김 (약한 거부)
#   2 = "즉시 스킵"  : 10초 이하 → 바로 넘김 (강한 거부)
#
# 판단 근거:
#   - 10초: 보통 전주/인트로가 끝나는 시점. 이 안에 넘기면 "아예 안 들은 것"
#   - 30초: 업계 표준 스트리밍 카운트 기준. 이보다 짧으면 "스킵"
# ================================================
def classify_skip(gap):
    if pd.isna(gap) or gap > 30:
        return 0  # 완주
    elif gap > 10:
        return 1  # 샘플 스킵
    else:
        return 2  # 즉시 스킵

df['skip_type'] = df['time_gap_seconds'].apply(classify_skip)

# 기존 is_skipped (호환성 유지): 30초 이하면 스킵으로 간주
df['is_skipped'] = (df['skip_type'] > 0).astype(int)

# ================================================
# 3. Familiarity: 완주(0)한 경우만 카운트
# ================================================
df['valid_play'] = (df['skip_type'] == 0).astype(int)
df['familiarity'] = (df.groupby('song_id')['valid_play']
                       .apply(lambda x: x.shift().cumsum().fillna(0))
                       .reset_index(level=0, drop=True)
                       .astype(int))
df = df.drop(columns=['valid_play'])

print(f"Total rows after parsing: {len(df)}")

# ================================================
# 4. Relisten within 7 days
# ================================================
df['next_song_timestamp'] = df.groupby('song_id')['timestamp'].shift(-1)
df['relisten_within_7d'] = ((df['next_song_timestamp'] - df['timestamp']).dt.days <= 7).astype(int)

# ================================================
# 5. Satisfaction Score 고도화
#
# 기존: (1 - is_skipped) + relisten_within_7d  → 0~2점 단순 합산
#
# 신규: skip_type을 반영한 세분화된 점수
#   완주(0)      = 1.0점  (정상 감상)
#   샘플 스킵(1) = 0.3점  (일단 들어봤으니 아예 0은 아님)
#   즉시 스킵(2) = 0.0점  (완전 거부)
#   + relisten_within_7d = 0.0~1.0점 추가
#   → 총 0.0 ~ 2.0점 범위 유지
# ================================================
skip_score_map = {0: 1.0, 1: 0.3, 2: 0.0}
df['listen_score'] = df['skip_type'].map(skip_score_map)
df['satisfaction_score'] = df['listen_score'] + df['relisten_within_7d']

df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print("Feature engineering complete.")

# ================================================
# 6. 스킵 타이밍 분포 확인
# ================================================
print("\n=== 스킵 타이밍 분포 ===")
counts = df['skip_type'].value_counts().sort_index()
total = len(df)
labels = {0: '완주 (30초+)', 1: '샘플 스킵 (10~30초)', 2: '즉시 스킵 (0~10초)'}
for st, cnt in counts.items():
    print(f"  {labels[st]}: {cnt}건 ({cnt/total*100:.1f}%)")

print("\n=== 스킵 타입별 평균 satisfaction_score ===")
print(df.groupby('skip_type')['satisfaction_score'].mean().rename(labels))

print("\n=== 재청취율 비교 (완주 vs 샘플스킵 vs 즉시스킵) ===")
print(df.groupby('skip_type')['relisten_within_7d'].mean().rename(labels))

print("\n--- TOP 5 SONGS (완주 기준) ---")
valid_df = df[df['skip_type'] == 0]
print(valid_df['song_id'].value_counts().head(5))
