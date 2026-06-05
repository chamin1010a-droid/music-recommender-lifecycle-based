import pandas as pd
import numpy as np
import codecs, sys

sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

# ======================================================
# 한국어 제목 ↔ 영어 제목 대응 테이블
# (YouTube Music 기록에는 영어 제목으로 저장됨)
# ======================================================
KOREAN_TO_ENGLISH = {
    # 검정치마 THIRSTY 앨범
    '빨간 나를': 'Holiday',
    '빨간나를': 'Holiday',
    '틀린질문': 'Wrong Question',
    '섬': 'Island (queen of diamonds)',
    '상수역': 'Sangsu station',
    '광견일기': 'Mad dog diary',
    '하와이 검은 모래': 'Hawaiian black sand',
    '맑고 묽게': 'Thinner than water',
    '그늘은 그림자로': 'My shadow',
    '피와 갈증': 'Blood and thirst (king of hurts)',

    # 검정치마 TEEN TROUBLES 앨범
    '불세례': 'Baptized In Fire',
    '어린양': 'My Little Lambs',
    '매미들': 'Cicadas',
    '따라갈래': 'Follow You',
    '미는 남자': 'Min',

    # 한로로 (HANRORO)
    '해초': '해초',  # 실제로 한글 그대로 저장됨
    '자처': '자처',
    '금붕어': '금붕어',
    '정류장': '정류장',
    '입춘': '입춘',
    '비틀비틀 짝짜꿍': '비틀비틀 짝짜꿍',
    '사랑하게 될 거야': '사랑하게 될 거야',
}

ENGLISH_TO_KOREAN = {v: k for k, v in KOREAN_TO_ENGLISH.items()}

print("=== 한/영 제목 대응 확인 ===")
print("\n[검정치마 THIRSTY 앨범]")
print("  빨간 나를 → Holiday (실제 데이터: 37회)")
print("  틀린질문  → Wrong Question (실제 데이터: 30회)")
print("  섬        → Island (queen of diamonds) (실제 데이터: 64회)")
print("  상수역    → Sangsu station (실제 데이터: 39회)")
print("  광견일기  → Mad dog diary (실제 데이터: 35회)")
print("  하와이 검은 모래 → Hawaiian black sand (실제 데이터: 44회)")
print("  맑고 묽게 → Thinner than water (실제 데이터: 25회)")
print("  그늘은 그림자로 → My shadow (실제 데이터: 24회)")
print("  피와 갈증 → Blood and thirst (king of hurts) (실제 데이터: 21회)")

# ======================================================
# 버그 수정: song_id 집계 오류 확인 및 진단
# 문제: generate_filtered_playlist.py에서 title 기반으로 검색 후
#       song_id 기반으로 카운트할 때 불일치 발생
# ======================================================
df = pd.read_csv(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("\n\n=== 버그 진단: 해초 집계 확인 ===")
haechu = df[df['title'] == '해초']
haechu_by_songid = df[df['song_id'].str.contains('해초', na=False)]
print(f"  title='해초' 검색 결과: {len(haechu)}건")
print(f"  song_id 포함 '해초' 검색 결과: {len(haechu_by_songid)}건")
print(f"  고유 song_id: {haechu['song_id'].unique()}")

# 실제 play count by song_id
song_total = df.groupby('song_id').size()
print(f"\n  해초 song_id로 집계한 총 재생수: {song_total.get('해초 - HANRORO - Topic', '없음')}")
print(f"  is_skipped=0 (유효 재생): {len(haechu[haechu['is_skipped']==0])}건")

# ======================================================
# 버그 원인 확인: new_candidate 분류 임계값 문제
# "3회 미만"은 너무 엄격함 → full song_id aggregation으로 재확인
# ======================================================
print("\n=== 버그 원인: generate_filtered_playlist.py의 분류 로직 확인 ===")
# 기존 코드는 df['song_id'].unique()로 순회하면서 각 song_id의 len을 세는데
# 문제는 parse 과정에서 동일 곡이 다른 song_id로 들어온 경우가 있을 수 있음

# 의심가는 케이스: 빈 아티스트 또는 다른 아티스트로 같은 곡이 저장된 경우
print("\n  '해초'가 포함된 모든 행:")
print(df[df['song_id'].str.contains('해초', na=False)][['title', 'artist', 'song_id', 'is_skipped']].head(10))

print("\n  'Holiday'가 포함된 모든 행 (빨간 나를):")
print(df[df['song_id'].str.contains('Holiday', na=False)][['title', 'artist', 'song_id', 'is_skipped']].head(10))

# 확인: song_id 기준 실제 총 카운트
print("\n=== 올바른 플레이리스트 생성을 위한 참고 정보 ===")
print("\n[검정치마 곡 전체 - 한국어 제목 대응]")
bsk_songs = df[df['artist'].str.contains('Black Skirts', na=False, case=False)]
for title, count in bsk_songs['title'].value_counts().items():
    kor = ENGLISH_TO_KOREAN.get(title, '한글제목없음(영어전용곡)')
    print(f"  {title} = 한국어: '{kor}' ({count}회)")
