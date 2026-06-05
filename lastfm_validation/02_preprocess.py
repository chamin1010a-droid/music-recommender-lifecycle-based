"""
Step 2: 전처리 + 세 가설 검증을 위한 피처 엔지니어링
- 사용자-곡별 누적 재생 횟수 (familiarity)
- 재청취 프록시(7일/3일)
- 스킵 프록시 (재생 간격 기반)
"""
import os
import pandas as pd
import numpy as np
from datetime import timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_FILE = os.path.join(DATA_DIR, "lastfm_1k.parquet")
FEATURES_FILE = os.path.join(DATA_DIR, "lastfm_1k_features.parquet")


def load_data():
    print("Loading parquet...")
    df = pd.read_parquet(CACHE_FILE)
    print(f"  Loaded: {len(df):,} rows")
    return df


def preprocess(df):
    """핵심 전처리"""
    print("\n[1/4] 정렬 및 필터링...")
    # 필수 컬럼만 + 결측 트랙 제거
    cols = ['user_id', 'timestamp', 'artist_name', 'track_name']
    df = df[cols].dropna(subset=['track_name']).copy()
    
    # 사용자+시간 순 정렬
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    # song_key: 아티스트 + 트랙 결합 (고유 곡 식별)
    df['song_key'] = df['artist_name'] + ' - ' + df['track_name']
    
    print(f"  유효 재생: {len(df):,} rows")
    return df


def add_familiarity(df):
    """사용자-곡별 누적 재생 횟수 (familiarity) 계산"""
    print("\n[2/4] Familiarity (누적 재생 횟수) 계산...")
    # 각 재생 이벤트가 해당 사용자가 해당 곡을 몇 번째 듣는 것인지
    df['play_seq'] = df.groupby(['user_id', 'song_key']).cumcount()  # 0-indexed
    df['familiarity'] = df['play_seq']  # i번째 재생 시점의 과거 누적 = i (0-indexed이므로 그대로)
    
    print(f"  완료. familiarity 범위: {df['familiarity'].min()} ~ {df['familiarity'].max()}")
    return df


def add_relisten_proxies(df):
    """재청취 프록시 계산: 7일/3일 이내 동일 곡 재생 여부"""
    print("\n[3/4] 재청취 프록시 계산 (7일/3일)...")
    
    # 같은 사용자-곡의 다음 재생 시각
    df['next_same_song_ts'] = df.groupby(['user_id', 'song_key'])['timestamp'].shift(-1)
    
    # 다음 재생까지의 간격 (일 단위)
    df['days_to_next'] = (df['next_same_song_ts'] - df['timestamp']).dt.total_seconds() / 86400
    
    # 재청취 프록시
    df['relisten_7d'] = (df['days_to_next'] <= 7).astype(int)
    df['relisten_3d'] = (df['days_to_next'] <= 3).astype(int)
    
    # 마지막 재생(다음 재생 없음)은 NaN → 0으로 (재청취 안 함)
    df['relisten_7d'] = df['relisten_7d'].fillna(0).astype(int)
    df['relisten_3d'] = df['relisten_3d'].fillna(0).astype(int)
    
    print(f"  relisten_7d 분포:\n{df['relisten_7d'].value_counts().to_string()}")
    print(f"  relisten_3d 분포:\n{df['relisten_3d'].value_counts().to_string()}")
    return df


def add_skip_proxy(df):
    """스킵 프록시: 다음 곡까지의 시간 간격으로 역추론"""
    print("\n[4/4] 스킵 프록시 계산 (재생 간격 기반)...")
    
    # 같은 사용자의 '바로 다음 재생'까지의 간격 (초)
    df['next_any_ts'] = df.groupby('user_id')['timestamp'].shift(-1)
    df['gap_seconds'] = (df['next_any_ts'] - df['timestamp']).dt.total_seconds()
    
    # 스킵 판정
    # gap < 30초: 즉시 스킵
    # 30초 <= gap < 60초: 스킵 추정  
    # gap >= 180초(3분): 완청 가능성 높음
    # 60초 ~ 180초: 불확실 (짧은 곡일 수 있음) → 완청으로 처리
    df['is_skip_proxy'] = 0
    df.loc[df['gap_seconds'] < 60, 'is_skip_proxy'] = 1  # 60초 미만 = 스킵
    
    # 세션 경계 (gap > 30분) → 스킵 아님 (세션이 끝난 것)
    df.loc[df['gap_seconds'] > 1800, 'is_skip_proxy'] = 0
    # 마지막 재생 (gap_seconds NaN) → 스킵 아님
    df.loc[df['gap_seconds'].isna(), 'is_skip_proxy'] = 0
    
    skip_rate = df['is_skip_proxy'].mean()
    print(f"  전체 추정 스킵률: {skip_rate:.1%}")
    print(f"  is_skip_proxy 분포:\n{df['is_skip_proxy'].value_counts().to_string()}")
    
    return df


def save_features(df):
    """피처 엔지니어링 결과 저장"""
    # 중간 계산용 컬럼 제거
    drop_cols = ['next_same_song_ts', 'next_any_ts']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    df.to_parquet(FEATURES_FILE, index=False)
    size_mb = os.path.getsize(FEATURES_FILE) / (1024*1024)
    print(f"\nSaved: {FEATURES_FILE} ({size_mb:.1f} MB)")
    return df


if __name__ == "__main__":
    df = load_data()
    df = preprocess(df)
    df = add_familiarity(df)
    df = add_relisten_proxies(df)
    df = add_skip_proxy(df)
    df = save_features(df)
    
    print("\n" + "="*60)
    print("Feature Engineering 완료!")
    print("="*60)
    print(f"  최종 shape: {df.shape}")
    print(f"  컬럼: {list(df.columns)}")
