"""
Step 1: Last.fm-1K 데이터셋 다운로드 및 기본 EDA
- Hugging Face에서 Parquet 형식으로 로드
- 기본 통계 확인 (사용자 수, 재생 수, 기간 범위)
- 로컬 캐시로 저장
"""
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_FILE = os.path.join(DATA_DIR, "lastfm_1k.parquet")

def download_dataset():
    """Hugging Face에서 Last.fm-1K 데이터셋을 다운로드하여 Parquet로 저장"""
    if os.path.exists(CACHE_FILE):
        print(f"✅ 캐시 파일이 이미 존재합니다: {CACHE_FILE}")
        return pd.read_parquet(CACHE_FILE)
    
    print("📥 Hugging Face에서 Last.fm-1K 데이터셋 다운로드 중...")
    from datasets import load_dataset
    
    ds = load_dataset("matthewfranglen/lastfm-1k", split="train")
    df = ds.to_pandas()
    
    # 로컬 캐시 저장
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_parquet(CACHE_FILE, index=False)
    print(f"💾 저장 완료: {CACHE_FILE} ({os.path.getsize(CACHE_FILE)/(1024*1024):.1f} MB)")
    
    return df


def basic_eda(df):
    """기본 탐색적 데이터 분석"""
    print("\n" + "="*60)
    print("📊 Last.fm-1K 기본 EDA")
    print("="*60)
    
    print(f"\n📋 데이터 shape: {df.shape}")
    print(f"\n📋 컬럼: {list(df.columns)}")
    print(f"\n📋 데이터 타입:")
    print(df.dtypes)
    
    print(f"\n📋 처음 5행:")
    print(df.head())
    
    print(f"\n📋 결측치:")
    print(df.isnull().sum())
    
    # 핵심 통계
    n_users = df['userid'].nunique() if 'userid' in df.columns else 'N/A'
    n_tracks = None
    
    # 컬럼명 확인 후 통계
    print(f"\n{'='*60}")
    print(f"🔑 핵심 통계")
    print(f"{'='*60}")
    print(f"  사용자 수: {n_users}")
    print(f"  총 재생 수: {len(df):,}")
    
    # 아티스트/트랙 관련 컬럼 탐색
    for col in df.columns:
        col_lower = col.lower()
        if 'art' in col_lower and 'name' in col_lower:
            print(f"  고유 아티스트 수: {df[col].nunique():,}")
        if 'tra' in col_lower and 'name' in col_lower:
            print(f"  고유 트랙 수: {df[col].nunique():,}")
        if 'time' in col_lower or 'date' in col_lower:
            print(f"  기간 범위: {df[col].min()} ~ {df[col].max()}")
    
    # 사용자별 재생 수 분포
    if 'userid' in df.columns:
        user_plays = df.groupby('userid').size()
        print(f"\n📊 사용자별 재생 수 분포:")
        print(f"  평균: {user_plays.mean():,.0f}")
        print(f"  중앙값: {user_plays.median():,.0f}")
        print(f"  최소: {user_plays.min():,}")
        print(f"  최대: {user_plays.max():,}")
        print(f"  Q1(25%): {user_plays.quantile(0.25):,.0f}")
        print(f"  Q3(75%): {user_plays.quantile(0.75):,.0f}")


if __name__ == "__main__":
    df = download_dataset()
    basic_eda(df)
