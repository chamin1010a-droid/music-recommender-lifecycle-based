"""
음원 생애주기(Song Lifecycle) 기반 추천 알고리즘
==============================================
핵심 설계 원칙:
1. 2-Layer 구조: 아티스트 Tier → 곡별 Temperature
2. 비대칭 흐름: 성장은 Bottom-Up(곡→아티스트), 쇠퇴는 Top-Down(아티스트→곡)
3. 재생 빈도의 '감소 추세(기울기)'도 온도 판별의 핵심 신호
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from datetime import datetime, timedelta
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')


# =============================================================================
# Layer 0: 아티스트명 정규화 (Artist Name Normalizer)
# =============================================================================
class ArtistNameNormalizer:
    """
    유튜브 뮤직 데이터에서 동일 아티스트가 여러 이름으로 분산되는 문제를 해결한다.
    예: 'Oasis - Topic', 'OasisVEVO', 'Oasis' → 모두 'Oasis - Topic'으로 통합

    두 단계로 작동:
    1. 자동 정규화: '- Topic' 접미사, 'VEVO' 등 패턴 기반
    2. 수동 매핑: 한글↔영문 등 자동으로 잡을 수 없는 케이스
    """

    # 한글명 → 영문 Topic 이름 수동 매핑
    MANUAL_ALIASES = {
        '잔나비': 'JANNABI - Topic',
        '검정치마': 'The Black Skirts - Topic',
        '한로로 HANRORO': 'HANRORO - Topic',
        '카더가든 (Car, the garden)': 'Car, the Garden - Topic',
        '장범준': 'Jang Beom June - Topic',
        '로이킴 Roy Kim': 'Roy Kim - Topic',
        '쏜애플 / THORNAPPLE': 'Thornapple - Topic',
        '노을Noel': 'Noel - Topic',
        '웅키': 'Ungki - Topic',
    }

    # 무시할 채널 (음악 아티스트가 아닌 방송/플랫폼 채널)
    # 이들은 곡 제목에서 실제 아티스트를 추출해야 하지만, 지금은 그대로 둠
    BROADCAST_CHANNELS = {
        'KBS Kpop', 'Beginagain 비긴어게인', '1theK (원더케이)',
        'Stone Music Entertainment', 'Mnet TV', 'BBC Music',
        'Like It Music', '미러볼 뮤직 - Mirrorball Music',
    }

    def __init__(self):
        self.alias_map = {}  # original_name → canonical_name
        self.merge_stats = {}  # canonical_name → count of merged names

    def build_alias_map(self, artist_names):
        """주어진 아티스트 이름 목록에서 정규화 매핑을 구축한다."""
        topic_artists = {}  # base_name → 'XXX - Topic' 전체 이름

        # Step 1: '- Topic' 아티스트를 기준으로 base name 사전 구축
        for name in artist_names:
            if isinstance(name, str) and name.endswith(' - Topic'):
                base = name.replace(' - Topic', '').strip().lower()
                topic_artists[base] = name

        # Step 2: 모든 아티스트에 대해 매핑 생성
        for name in artist_names:
            if not isinstance(name, str):
                continue

            # 이미 Topic 형태면 자기 자신으로 매핑
            if name.endswith(' - Topic'):
                self.alias_map[name] = name
                continue

            # 수동 매핑 확인
            if name in self.MANUAL_ALIASES:
                canonical = self.MANUAL_ALIASES[name]
                self.alias_map[name] = canonical
                continue

            # 자동 매핑: VEVO 제거 후 Topic 버전 찾기
            cleaned = name.replace('VEVO', '').replace('vevo', '').strip().lower()
            if cleaned in topic_artists:
                self.alias_map[name] = topic_artists[cleaned]
                continue

            # 매칭 안 되면 자기 자신 유지
            self.alias_map[name] = name

        # 병합 통계 계산
        canonical_counts = defaultdict(list)
        for orig, canonical in self.alias_map.items():
            if orig != canonical:
                canonical_counts[canonical].append(orig)
        self.merge_stats = canonical_counts

        return self.alias_map

    def normalize_dataframe(self, df):
        """DataFrame의 artist 컬럼을 정규화한다."""
        if not self.alias_map:
            self.build_alias_map(df['artist'].unique())

        df = df.copy()
        df['artist_original'] = df['artist']
        df['artist'] = df['artist'].map(lambda x: self.alias_map.get(x, x))
        return df

    def summary(self):
        """정규화 결과를 출력한다."""
        if not self.merge_stats:
            print("  (정규화된 아티스트 없음)")
            return

        print(f"\n🔗 아티스트명 정규화: {len(self.merge_stats)}건 병합")
        print("-" * 50)
        for canonical, originals in self.merge_stats.items():
            print(f"  {canonical}")
            for orig in originals:
                print(f"    ← {orig}")


# =============================================================================
# Layer 1: 아티스트 친밀도 등급 (Artist Affinity Tier)
# =============================================================================
class ArtistTierClassifier:
    """
    사용자의 청취 로그를 아티스트 단위로 클러스터링하여
    Tier S(생태계) / A(히트곡) / B(원히트)로 자동 분류.
    기준값은 하드코딩하지 않고 K-Means가 데이터로부터 결정.
    """

    def __init__(self, df, reference_date=None):
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['timestamp'])
        self.reference_date = reference_date or self.df['date'].max()
        self.artist_stats = None
        self.tier_map = {}  # artist -> tier (S, A, B)

    def compute_artist_stats(self):
        """아티스트별 핵심 통계를 계산한다."""
        recent_30d = self.reference_date - timedelta(days=30)

        stats = []
        for artist, group in self.df.groupby('artist'):
            unique_songs = group['song_id'].nunique()
            total_plays = len(group)
            recent_plays = len(group[group['date'] >= recent_30d])

            # 청취 기간 (첫 재생 ~ 마지막 재생)
            first_play = group['date'].min()
            last_play = group['date'].max()
            listening_span_days = max((last_play - first_play).days, 1)

            # 재생 밀도 (일 평균 재생 횟수)
            play_density = total_plays / listening_span_days

            # 최근 활동 여부 (마지막 재생으로부터의 경과일)
            days_since_last = (self.reference_date - last_play).days

            stats.append({
                'artist': artist,
                'unique_songs': unique_songs,
                'total_plays': total_plays,
                'recent_plays_30d': recent_plays,
                'listening_span_days': listening_span_days,
                'play_density': play_density,
                'days_since_last': days_since_last
            })

        self.artist_stats = pd.DataFrame(stats)
        return self.artist_stats

    def classify_tiers(self):
        """K-Means 클러스터링으로 Tier를 자동 분류한다."""
        if self.artist_stats is None:
            self.compute_artist_stats()

        # 클러스터링에 사용할 피처: 고유 곡 수, 총 재생 횟수, 최근 30일 재생
        features = self.artist_stats[['unique_songs', 'total_plays', 'recent_plays_30d']].values

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # K=3 클러스터로 분류
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_scaled)
        self.artist_stats['cluster'] = clusters

        # 클러스터 중심의 총 재생 횟수 기준으로 Tier 매핑
        # 가장 높은 총재생 → S, 중간 → A, 가장 낮은 → B
        cluster_means = self.artist_stats.groupby('cluster')['total_plays'].mean()
        sorted_clusters = cluster_means.sort_values(ascending=False).index.tolist()

        tier_labels = {sorted_clusters[0]: 'S', sorted_clusters[1]: 'A', sorted_clusters[2]: 'B'}
        self.artist_stats['tier'] = self.artist_stats['cluster'].map(tier_labels)

        # tier_map 생성
        self.tier_map = dict(zip(self.artist_stats['artist'], self.artist_stats['tier']))

        return self.tier_map

    def get_tier(self, artist):
        return self.tier_map.get(artist, 'B')

    def summary(self):
        """Tier별 요약을 출력한다."""
        if self.artist_stats is None:
            return

        print("\n" + "=" * 60)
        print("📊 아티스트 Tier 분류 결과")
        print("=" * 60)

        for tier in ['S', 'A', 'B']:
            tier_df = self.artist_stats[self.artist_stats['tier'] == tier]
            tier_names = {
                'S': '🏆 Tier S (생태계 아티스트)',
                'A': '⭐ Tier A (히트곡 아티스트)',
                'B': '🎵 Tier B (원히트 아티스트)'
            }
            print(f"\n{tier_names[tier]} — {len(tier_df)}명")
            print("-" * 50)
            top = tier_df.nlargest(5, 'total_plays')
            for _, row in top.iterrows():
                print(f"  {row['artist'][:30]:<32} "
                      f"곡 {row['unique_songs']:>3}개 | "
                      f"재생 {row['total_plays']:>4}회 | "
                      f"최근30일 {row['recent_plays_30d']:>3}회")

            if len(tier_df) > 5:
                print(f"  ...외 {len(tier_df) - 5}명 더")


# =============================================================================
# Layer 2: 곡별 연속 점수 (Song Scorer) — v2
# =============================================================================
class SongScorer:
    """
    [v2] 이산적 라벨(Rising/Warm/Cool/Frozen) 대신 연속 점수로 곡을 평가한다.

    두 개의 독립 축:
      1. 호감도 (Affinity, 0~1): 이 곡을 얼마나 좋아하는가? (느리게 변함)
      2. 모멘텀 (Momentum, 0~1): 요즘 이 곡/가수를 듣고 있는가? (빠르게 변함)

    세 번째 축(맥락 적합도)은 플레이리스트 생성 시점에 동적 계산한다.

    핵심 장점:
      - 경계선 문제 없음 (연속 점수)
      - 보정/쉴드 로직 불필요 (각 축이 독립)
      - Cool 탈출 문제 없음 (한 번 들으면 모멘텀 즉시 상승)
    """

    # 호감도 가중치
    AFFINITY_WEIGHTS = {
        'skip_quality': 0.30,      # 1 - skip_rate
        'trajectory': 0.20,         # 전반skip → 후반skip 변화 (연속)
        'play_depth': 0.20,         # log 재생횟수 (정규화)
        'proactive': 0.15,          # 능동성 점수
        'artist_familiarity': 0.15, # 이 가수의 곡을 얼마나 아는가
    }

    # 모멘텀 가중치
    MOMENTUM_WEIGHTS = {
        'recency': 0.25,            # exp(-days/60) 감쇠 (완만한 감쇠)
        'recent_freq': 0.20,        # 최근 30일 재생 빈도
        'artist_trend': 0.20,       # 아티스트 주간 추세
        'momentum_delta': 0.35,     # 가속도: 최근 30일 vs 이전 60일 비교
    }

    def __init__(self, df, tier_map, reference_date=None, metadata_path=None,
                 peak_age_range=(13, 24), user_birth_year=1998):
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['timestamp'])
        self.tier_map = tier_map
        self.reference_date = reference_date or self.df['date'].max()
        self.song_scores = {}   # song_id -> {affinity, momentum, ...메타데이터}

        # 전역 통계 (정규화용)
        self._max_plays = self.df.groupby('song_id').size().max()
        self._user_daily_avg = len(self.df) / max(
            (self.df['date'].max() - self.df['date'].min()).days, 1)

        # 아티스트별 곡 수 (인지율 계산용)
        self._artist_song_counts = self.df.groupby('artist')['song_id'].nunique().to_dict()

        # 아티스트 주간 추세 사전 계산
        self._artist_weekly_trends = self._compute_artist_weekly_trends()

        # 메타데이터 로드
        self.metadata = {}
        if metadata_path and __import__('os').path.exists(metadata_path):
            meta_df = pd.read_csv(metadata_path, encoding='utf-8-sig')
            if 'matched' in meta_df.columns:
                meta_df = meta_df[meta_df['matched'] == True]
            for _, row in meta_df.iterrows():
                try:
                    self.metadata[row['song_id']] = {
                        'release_year': float(row.get('release_year'))
                            if not pd.isna(row.get('release_year')) else None,
                        'genre': row.get('genre')
                    }
                except:
                    pass

    def _compute_artist_weekly_trends(self):
        """아티스트별 주간 재생 추세(최근4주 / 전체평균)를 사전 계산한다."""
        trends = {}
        recent_4w = self.reference_date - timedelta(days=28)

        for artist, group in self.df.groupby('artist'):
            if len(group) < 5:
                trends[artist] = 0.5  # 데이터 부족 → 중립
                continue

            group_sorted = group.sort_values('date')
            group_copy = group_sorted.copy()
            group_copy['week'] = (
                group_copy['date'].dt.isocalendar().week.astype(int) +
                group_copy['date'].dt.isocalendar().year.astype(int) * 52
            )
            weekly = group_copy.groupby('week').size()

            if len(weekly) < 2:
                trends[artist] = 0.5
                continue

            overall_avg = weekly.mean()
            recent_plays = len(group[group['date'] >= recent_4w])
            recent_weekly_avg = recent_plays / 4.0  # 4주 평균

            if overall_avg > 0:
                ratio = recent_weekly_avg / overall_avg
                trends[artist] = min(1.0, ratio)  # 1.0 캡
            else:
                trends[artist] = 0.0

        return trends

    def _compute_affinity(self, skip_rate, first_half_skip, second_half_skip,
                          total_plays, proactive_score, artist, song_id):
        """
        [축 1: 호감도] 이 곡을 얼마나 좋아하는가? (0.0 ~ 1.0)

        5개 구성 요소:
        1. 스킵 품질 (0~1): 1 - skip_rate
        2. 궤적 보너스 (0~1): 전반→후반 스킵률 변화 (연속, 경계선 없음)
        3. 재생 깊이 (0~1): log₂ 정규화된 재생 횟수
        4. 능동성 (0~1): 검색 후 재생 등 능동적 신호
        5. 아티스트 인지율 (0~1): 이 가수의 곡을 얼마나 많이 아는가
        """
        w = self.AFFINITY_WEIGHTS

        # 1. 스킵 품질
        skip_quality = 1.0 - skip_rate

        # 2. 궤적 보너스 (Zone 분류를 연속 변수로 녹인 것)
        if first_half_skip is not None and second_half_skip is not None:
            trajectory = np.clip(0.5 + (first_half_skip - second_half_skip), 0, 1)
        else:
            trajectory = 0.5  # 데이터 부족 → 중립

        # 3. 재생 깊이
        play_depth = np.log2(1 + total_plays) / np.log2(1 + self._max_plays)

        # 4. 능동성
        proactive = proactive_score

        # 5. 아티스트 인지율
        artist_total_songs = self._artist_song_counts.get(artist, 1)
        # 이 가수의 곡 중 3회 이상 들은 곡 수
        artist_songs_df = self.df[self.df['artist'] == artist]
        songs_known = artist_songs_df.groupby('song_id').size()
        songs_known_count = (songs_known >= 3).sum()
        artist_familiarity = min(1.0, songs_known_count / max(artist_total_songs, 1))

        affinity = (
            w['skip_quality'] * skip_quality
            + w['trajectory'] * trajectory
            + w['play_depth'] * play_depth
            + w['proactive'] * proactive
            + w['artist_familiarity'] * artist_familiarity
        )

        return round(np.clip(affinity, 0, 1), 3)

    def _compute_momentum(self, days_since_last, recent_30d_plays, artist,
                          prev_60d_plays=0):
        """
        [축 2: 모멘텀] 요즘 이 곡/가수를 듣고 있는가? (0.0 ~ 1.0)

        4개 구성 요소:
        1. 리센시 (0~1): exp(-days/60) 감쇠 (완만한 감쇠, 7일 유예)
        2. 최근 빈도 (0~1): 최근 30일 재생 / 기대 재생
        3. 아티스트 추세 (0~1): 최근 4주 vs 전체 평균
        4. 가속도 (0~1): 최근 30일 빈도 vs 이전 30~90일 빈도 비교
           → "요즘 다시 듣기 시작한 곡"을 포착
        """
        w = self.MOMENTUM_WEIGHTS

        # 1. 리센시 — 7일 유예 후 완만한 지수 감쇠 (반감기 ~42일)
        effective_days = max(0, days_since_last - 7)
        recency = np.exp(-effective_days / 60.0)

        # 2. 최근 빈도
        expected_30d = self._user_daily_avg * 30 / max(
            self.df['song_id'].nunique(), 1) * 5  # 곡당 기대 재생
        recent_freq = min(1.0, recent_30d_plays / max(expected_30d, 0.5))

        # 3. 아티스트 추세
        artist_trend = self._artist_weekly_trends.get(artist, 0.5)

        # 4. 가속도(Δ) — 최근 30일 재생 vs 이전 30~90일(60일간) 평균 월간 재생
        #    prev_60d_plays: 31일~90일 전 사이의 재생 횟수 (60일 구간)
        #    이전 월간 평균 = prev_60d_plays / 2
        prev_monthly_avg = prev_60d_plays / 2.0
        if prev_monthly_avg > 0:
            # 최근이 이전보다 얼마나 증가했나 (비율)
            acceleration_ratio = recent_30d_plays / prev_monthly_avg
            # 1.0 = 동일, >1 = 상승세, <1 = 하락세
            # 0~1로 정규화: ratio 2.0 이상이면 1.0 (큰 상승)
            momentum_delta = np.clip((acceleration_ratio - 0.5) / 1.5, 0, 1)
        elif recent_30d_plays > 0:
            # 이전에 전혀 안 듣다가 최근 듣기 시작 → 큰 가속도
            momentum_delta = 1.0
        else:
            # 이전에도 안 듣고 지금도 안 들음 → 중립
            momentum_delta = 0.3

        momentum = (
            w['recency'] * recency
            + w['recent_freq'] * recent_freq
            + w['artist_trend'] * artist_trend
            + w['momentum_delta'] * momentum_delta
        )

        return round(np.clip(momentum, 0, 1), 3)

    def _get_zone_label(self, first_half_skip, second_half_skip, skip_rate,
                        total_plays):
        """
        [표시용만] Zone 라벨을 계산한다. 점수에는 영향 없음.
        사용자에게 "이 곡은 듣다보니 좋아지는 중이에요" 같은 설명을 할 때 사용.
        """
        if total_plays < 3 or first_half_skip is None:
            return 'N/A'

        if first_half_skip <= 0.33 and skip_rate <= 0.3:
            return '처음부터 좋아함 ♥'
        elif first_half_skip > second_half_skip + 0.10:
            return '듣다보니 좋아짐 ↑'
        elif second_half_skip > first_half_skip + 0.10:
            return '좋다가 질림 ↓'
        else:
            return '그냥저냥 ~'

    def score_all_songs(self):
        """모든 곡의 호감도와 모멘텀 점수를 산출한다."""

        # 곡별 proactive_score 집계
        song_proactive = {}
        if 'proactive_score' in self.df.columns:
            proactive_agg = self.df.groupby('song_id').agg(
                avg_proactive=('proactive_score', 'mean'),
                proactive_plays=('is_proactive', 'sum')
            )
            for sid, row in proactive_agg.iterrows():
                song_proactive[sid] = round(row['avg_proactive'], 3)

        song_groups = self.df.groupby('song_id')

        for song_id, group in song_groups:
            artist = group['artist'].iloc[0]
            if pd.isna(artist):
                artist = 'Unknown'
            title = group['title'].iloc[0]
            if pd.isna(title):
                title = 'Unknown'
            tier = self.tier_map.get(artist, 'B')

            total_plays = len(group)
            total_skips = group['is_skipped'].sum() if 'is_skipped' in group.columns else 0
            skip_rate = total_skips / total_plays if total_plays > 0 else 0

            last_play = group['date'].max()
            days_since_last = (self.reference_date - last_play).days

            # 최근 30일 재생
            recent_30d_mask = group['date'] >= (self.reference_date - timedelta(days=30))
            recent_30d_plays = len(group[recent_30d_mask])

            # 이전 30~90일 재생 (가속도 계산용)
            prev_start = self.reference_date - timedelta(days=90)
            prev_end = self.reference_date - timedelta(days=30)
            prev_60d_mask = (group['date'] >= prev_start) & (group['date'] < prev_end)
            prev_60d_plays = len(group[prev_60d_mask])

            # 전/후반 스킵률
            group_sorted = group.sort_values('date')
            if total_plays >= 3:
                half = total_plays // 2
                first_half_skip = group_sorted.iloc[:half]['is_skipped'].mean()
                second_half_skip = group_sorted.iloc[half:]['is_skipped'].mean()
            else:
                first_half_skip = None
                second_half_skip = None

            # 능동성 점수
            proactive_score = song_proactive.get(song_id, 0.0)

            # 시간대 계산
            if 'hour' not in group.columns:
                hours = pd.to_datetime(group['timestamp']).dt.hour
            else:
                hours = group['hour']

            def get_time_of_day(h):
                if 0 <= h < 6: return 'Night'
                elif 6 <= h < 12: return 'Morning'
                elif 12 <= h < 18: return 'Afternoon'
                else: return 'Evening'

            time_counts = hours.apply(get_time_of_day).value_counts()
            peak_time = time_counts.index[0] if len(time_counts) > 0 else 'Unknown'

            # 메타데이터
            genre = None
            if song_id in self.metadata:
                genre = self.metadata[song_id].get('genre')

            # === 핵심: 연속 점수 계산 ===
            affinity = self._compute_affinity(
                skip_rate, first_half_skip, second_half_skip,
                total_plays, proactive_score, artist, song_id
            )
            momentum = self._compute_momentum(
                days_since_last, recent_30d_plays, artist,
                prev_60d_plays=prev_60d_plays
            )

            # Zone 라벨 (표시용)
            zone_label = self._get_zone_label(
                first_half_skip, second_half_skip, skip_rate, total_plays
            )

            self.song_scores[song_id] = {
                'song_id': song_id,
                'title': title,
                'artist': artist,
                'tier': tier,
                'affinity': affinity,
                'momentum': momentum,
                'zone_label': zone_label,
                'total_plays': total_plays,
                'skip_rate': round(skip_rate, 2),
                'first_half_skip': round(first_half_skip, 2) if first_half_skip is not None else None,
                'second_half_skip': round(second_half_skip, 2) if second_half_skip is not None else None,
                'days_since_last': days_since_last,
                'recent_30d_plays': recent_30d_plays,
                'peak_time_of_day': peak_time,
                'genre': genre,
                'avg_proactive': proactive_score,
            }

        return self.song_scores

    def summary(self):
        """점수 분포 요약을 출력한다."""
        print("\n" + "=" * 60)
        print("📊 곡별 점수 산출 결과 (v2: 연속 점수 체계)")
        print("=" * 60)

        scores_df = pd.DataFrame(self.song_scores.values())

        # 대시보드: 점수 분포
        print(f"\n총 {len(scores_df)}곡 분석 완료")
        print(f"  호감도(Affinity)  — 평균 {scores_df['affinity'].mean():.2f}, "
              f"중앙값 {scores_df['affinity'].median():.2f}")
        print(f"  모멘텀(Momentum)  — 평균 {scores_df['momentum'].mean():.2f}, "
              f"중앙값 {scores_df['momentum'].median():.2f}")

        # 호감도 TOP 10
        print("\n🔥 호감도 TOP 10")
        print("-" * 50)
        top_aff = scores_df.nlargest(10, 'affinity')
        for _, row in top_aff.iterrows():
            print(f"  {row['title'][:30]:<32} "
                  f"호감 {row['affinity']:.2f} | 모멘텀 {row['momentum']:.2f} | "
                  f"{row['zone_label']}")

        # 모멘텀 TOP 10
        print("\n⚡ 모멘텀 TOP 10")
        print("-" * 50)
        top_mom = scores_df.nlargest(10, 'momentum')
        for _, row in top_mom.iterrows():
            print(f"  {row['title'][:30]:<32} "
                  f"호감 {row['affinity']:.2f} | 모멘텀 {row['momentum']:.2f} | "
                  f"{row['zone_label']}")

        # 숨겨진 보석: 호감도 높은데 모멘텀 낮은 곡 (= 예전의 Cool/Frozen이었을 곡)
        hidden_gems = scores_df[(scores_df['affinity'] >= 0.4) & (scores_df['momentum'] <= 0.15)]
        if len(hidden_gems) > 0:
            print(f"\n💎 숨겨진 보석 (호감도 ≥ 0.4 + 모멘텀 ≤ 0.15) — {len(hidden_gems)}곡")
            print("-" * 50)
            for _, row in hidden_gems.nlargest(10, 'affinity').iterrows():
                print(f"  {row['title'][:30]:<32} "
                      f"호감 {row['affinity']:.2f} | 모멘텀 {row['momentum']:.2f} | "
                      f"{row['artist'][:15]}")




# 비대칭 흐름 감지기 (Asymmetric Flow Detector)
# =============================================================================
class AsymmetricFlowDetector:
    """
    성장과 쇠퇴의 비대칭 흐름을 감지한다.

    📈 성장 (Bottom-Up): 곡 1개의 재생이 급증 → 같은 아티스트의 다른 곡 추천 기회
    📉 쇠퇴 (Top-Down): 아티스트 전체의 재생이 감소 → 해당 아티스트의 모든 곡 일괄 냉각
    """

    def __init__(self, df, tier_map, reference_date=None):
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['timestamp'])
        self.tier_map = tier_map
        self.reference_date = reference_date or self.df['date'].max()
        self.growth_signals = []     # 성장 신호 감지된 아티스트 목록
        self.decline_signals = []    # 쇠퇴 신호 감지된 아티스트 목록

    def detect_growth_bottom_up(self):
        """
        📈 Bottom-Up 성장 감지:
        최근 14일 내에 특정 곡 1개의 재생이 폭발적으로 증가한 아티스트를 찾는다.
        → 이 아티스트의 아직 안 들어본 다른 곡을 추천할 기회.

        [정제] 최소 재생 5회 + 전체 30% 이상이어야 성장 판정
               (3회 기준은 과민 반응이 너무 심함)
        """
        recent_14d = self.reference_date - timedelta(days=14)
        recent_df = self.df[self.df['date'] >= recent_14d]

        self.growth_signals = []

        for artist, group in recent_df.groupby('artist'):
            # 최근 14일 내 가장 많이 들은 곡
            top_song = group['song_id'].value_counts()
            if len(top_song) == 0:
                continue

            top_song_id = top_song.index[0]
            top_song_recent_plays = top_song.iloc[0]

            # 이 곡의 과거 전체 재생 대비 최근 14일 비율이 높으면 → 성장 신호
            total_history = len(self.df[self.df['song_id'] == top_song_id])

            if total_history > 0 and top_song_recent_plays >= 5:  # 최소 5회 (3→5 상향)
                recent_ratio = top_song_recent_plays / total_history
                if recent_ratio >= 0.3:  # 전체 재생의 30% 이상이 최근 14일에 집중
                    all_songs_by_artist = self.df[self.df['artist'] == artist]['song_id'].nunique()

                    top_title = group[group['song_id'] == top_song_id]['title'].iloc[0]

                    self.growth_signals.append({
                        'artist': artist,
                        'trigger_song': top_title,
                        'trigger_song_id': top_song_id,
                        'recent_plays': top_song_recent_plays,
                        'recent_ratio': round(recent_ratio, 2),
                        'artist_total_songs': all_songs_by_artist,
                        'tier': self.tier_map.get(artist, 'B')
                    })

        return self.growth_signals

    def detect_decline_top_down(self):
        """
        📉 Top-Down 쇠퇴 감지:
        아티스트 전체의 주간 재생 추세가 하락하고 있는 아티스트를 찾는다.
        → 해당 아티스트의 모든 곡을 일괄 냉각 처리할 후보.

        [정제] 쇠퇴 심각도(severity) 3단계:
          - critical: 잔존율 30% 이하 + 기울기 급하락 → Warm→Cool 강등, Cool→Frozen 강등
          - severe:   잔존율 40% 이하 → Warm→Cool 강등
          - mild:     잔존율 50% 이하 → 로그만 남기고 모니터링 (강등 없음)
        """
        self.decline_signals = []

        for artist, group in self.df.groupby('artist'):
            if len(group) < 15:  # 최소 15회 재생 (10→15 상향)
                continue

            group = group.sort_values('date')
            group = group.copy()
            group['week'] = group['date'].dt.isocalendar().week.astype(int) + \
                            group['date'].dt.isocalendar().year.astype(int) * 52
            weekly = group.groupby('week').size()

            if len(weekly) < 6:  # 최소 6주 데이터 필요 (4→6 상향)
                continue

            # 후반부(최근 절반) vs 전반부 비교
            mid = len(weekly) // 2
            first_half_avg = weekly.iloc[:mid].mean()
            second_half_avg = weekly.iloc[mid:].mean()

            if first_half_avg < 1:  # 전반기 재생이 너무 적으면 스킵
                continue

            # 선형 기울기
            x = np.arange(len(weekly))
            y = weekly.values.astype(float)
            slope = np.polyfit(x, y, 1)[0]

            decline_ratio = second_half_avg / max(first_half_avg, 0.1)

            # 최근 30일 재생 횟수 확인
            recent_30d = len(group[group['date'] >= (self.reference_date - timedelta(days=30))])

            # 쇠퇴 심각도 판별
            severity = None
            if decline_ratio <= 0.30 and slope < -0.3:
                severity = 'critical'   # 잔존율 30% 이하 + 급하락
            elif decline_ratio <= 0.40 and slope < -0.2:
                severity = 'severe'     # 잔존율 40% 이하
            elif decline_ratio <= 0.50 and slope < -0.1:
                severity = 'mild'       # 잔존율 50% 이하 → 모니터링만

            if severity:
                self.decline_signals.append({
                    'artist': artist,
                    'severity': severity,
                    'slope': round(slope, 3),
                    'first_half_avg': round(first_half_avg, 1),
                    'second_half_avg': round(second_half_avg, 1),
                    'decline_ratio': round(decline_ratio, 2),
                    'recent_30d_plays': recent_30d,
                    'tier': self.tier_map.get(artist, 'B')
                })

        return self.decline_signals

    def apply_decline_correction(self, song_temps):
        """
        [핵심 신규 기능] 쇠퇴 감지 결과를 실제 곡 온도에 반영한다.

        이전: detect_decline_top_down()이 쇠퇴를 감지하지만 출력만 하고 버림
        이후: 쇠퇴 심각도에 따라 해당 아티스트의 곡 온도를 실제로 강등

        심각도별 처리:
          - critical: Warm→Cool, Cool→Frozen 강등
                      단, 곡 자체의 스킵률이 15% 이하인 '갓곡'은 보호 (Cool까지만)
          - severe:   Warm→Cool 강등만
                      단, 곡 자체의 스킵률이 20% 이하인 곡은 보호
          - mild:     강등 없음 (모니터링 단계)
        """
        decline_artists = {sig['artist']: sig for sig in self.decline_signals}
        correction_log = []  # 보정 기록

        for song_id, info in song_temps.items():
            artist = info['artist']
            if artist not in decline_artists:
                continue

            sig = decline_artists[artist]
            severity = sig['severity']
            skip_rate = info.get('skip_rate', 1.0)
            old_temp = info['temperature']

            new_temp = old_temp  # 기본값: 변경 없음

            if severity == 'critical':
                # 갓곡 쉴드: 스킵률 15% 이하 곡은 Cool까지만 강등 (Frozen 방지)
                if old_temp == 'Warm':
                    new_temp = 'Cool'
                elif old_temp == 'Cool' and skip_rate > 0.15:
                    new_temp = 'Frozen'
                # Rising/Steady는 건드리지 않음 (현재 활발하게 듣고 있으므로)

            elif severity == 'severe':
                if old_temp == 'Warm' and skip_rate > 0.20:
                    new_temp = 'Cool'

            # mild는 강등 없음

            if new_temp != old_temp:
                info['temperature'] = new_temp
                correction_log.append({
                    'song_id': song_id,
                    'title': info['title'],
                    'artist': artist,
                    'old_temp': old_temp,
                    'new_temp': new_temp,
                    'severity': severity,
                    'decline_ratio': sig['decline_ratio'],
                })

        self.correction_log = correction_log
        return correction_log

    def summary(self):
        """비대칭 흐름 감지 결과를 출력한다."""
        print("\n" + "=" * 60)
        print("📈📉 비대칭 흐름 감지 결과")
        print("=" * 60)

        print(f"\n📈 성장 신호 (Bottom-Up) — {len(self.growth_signals)}건")
        print("-" * 50)
        for sig in self.growth_signals[:10]:
            print(f"  [{sig['tier']}] {sig['artist'][:25]:<27} "
                  f"← \"{sig['trigger_song'][:25]}\" "
                  f"(최근 {sig['recent_plays']}회, 비율 {sig['recent_ratio']:.0%})")

        severity_icons = {'critical': '🔴', 'severe': '🟠', 'mild': '🟡'}
        print(f"\n📉 쇠퇴 신호 (Top-Down) — {len(self.decline_signals)}건")
        print("-" * 50)
        for sig in self.decline_signals[:15]:
            icon = severity_icons.get(sig['severity'], '')
            print(f"  {icon} [{sig['tier']}] {sig['artist'][:25]:<27} "
                  f"주간 {sig['first_half_avg']:.1f}→{sig['second_half_avg']:.1f} "
                  f"(잔존 {sig['decline_ratio']:.0%}, {sig['severity']})")

        # 쇠퇴 보정 결과 출력
        if hasattr(self, 'correction_log') and self.correction_log:
            print(f"\n🔧 쇠퇴 보정 적용: {len(self.correction_log)}곡 강등됨")
            print("-" * 50)
            for log in self.correction_log[:10]:
                print(f"  {log['title'][:30]:<32} "
                      f"{log['old_temp']:>7} → {log['new_temp']:<7} "
                      f"({log['artist'][:15]}, {log['severity']})")
            if len(self.correction_log) > 10:
                print(f"  ...외 {len(self.correction_log) - 10}곡 더")


# =============================================================================
# 플레이리스트 믹서 (Playlist Mixer)
# =============================================================================
class PlaylistMixer:
    """
    각 온도(Temperature)의 곡들을 설정된 비율로 섞어
    최적의 추천 플레이리스트를 생성한다.

    기본 비율: 성장 30% / 안정 25% / 순환 25% / 신곡 20%
    Frozen 곡은 일상 추천에서 배제 (장르 매칭에서만 후순위)

    Discovery 슬롯 통합:
      - 내부 Discovery: 성장 신호 아티스트의 덜 들은 곡, Discovery_Candidate
      - 외부 Discovery: Last.fm 유사 아티스트 기반 진짜 새 곡 (ExternalDiscoveryEngine)
      - 두 소스를 비율에 맞게 혼합하여 하나의 Discovery 슬롯에 배치
    """

    # v2 프리셋: 연속 점수 지수 (affinity, momentum, context) 조정
    PRESETS = {
        'default': {'w_affinity': 1.0, 'w_momentum': 1.0, 'w_context': 1.0, 'discovery_ratio': 0.20},
        'explore': {'w_affinity': 0.7, 'w_momentum': 0.5, 'w_context': 1.5, 'discovery_ratio': 0.40},
        'comfort': {'w_affinity': 1.5, 'w_momentum': 0.7, 'w_context': 0.8, 'discovery_ratio': 0.15},
        'lgbm': {'w_lgbm': 0.45, 'w_context': 0.55, 'discovery_ratio': 0.20},
    }

    # Discovery 내부 비율: 내부(기존 곡 재발견) vs 외부(Last.fm 신곡)
    DISCOVERY_MIX = {
        'default': {'internal': 0.50, 'external': 0.50},
        'explore': {'internal': 0.30, 'external': 0.70},
        'comfort': {'internal': 0.70, 'external': 0.30},
        'lgbm': {'internal': 0.50, 'external': 0.50},
    }

    def __init__(self, song_scores, growth_signals=None, similarity_engine=None, external_engine=None, lgbm_scores=None):
        """
        song_scores: SongScorer.song_scores 딕셔너리 (affinity, momentum 포함)
        growth_signals: AsymmetricFlowDetector.growth_signals 리스트
        similarity_engine: TagSimilarityEngine 인스턴스
        external_engine: ExternalDiscoveryEngine 인스턴스
        lgbm_scores: LightGBM v4 예측 호감도 딕셔너리 {song_id: predicted_affinity}
        """
        self.song_temps = song_scores  # 하위 호환을 위해 song_temps 이름 유지
        self.growth_signals = growth_signals or []
        self.similarity_engine = similarity_engine
        self.external_engine = external_engine
        self.lgbm_scores = lgbm_scores or {}

    def _get_artist_similarity(self, target_artist, seed_artist):
        """
        아티스트 간 유사도(0~1) 반환.
        우선순위: Last.fm API → Multi-Signal 곡 평균 유사도
        """
        if target_artist == seed_artist:
            return 1.0

        # 캐시 확인
        if not hasattr(self, '_seed_artist_sim_cache'):
            self._seed_artist_sim_cache = {}

        if seed_artist not in self._seed_artist_sim_cache:
            sim_map = {}
            lastfm = None
            if hasattr(self, 'similarity_engine') and self.similarity_engine:
                lastfm = getattr(self.similarity_engine, 'lastfm', None)
            
            if lastfm:
                similar_list = lastfm.get_similar_artists(seed_artist, limit=50)
                for entry in similar_list:
                    name = entry.get('name', '').lower()
                    match = entry.get('match', 0)
                    sim_map[name] = float(match)
            else:
                # ★ Last.fm 캐시 파일에서 직접 로드 시도
                sim_map = self._load_artist_sim_from_cache(seed_artist)
            
            if not sim_map and hasattr(self, 'similarity_engine') and self.similarity_engine:
                # ★ Fallback: Multi-Signal 곡 유사도 기반 아티스트 유사도
                # 시드 아티스트의 대표곡 3개 선정
                seed_songs = sorted(
                    [s for s in self.song_temps.values() if s.get('artist') == seed_artist],
                    key=lambda x: x.get('total_plays', 0), reverse=True
                )[:3]
                seed_ids = [s['song_id'] for s in seed_songs]
                
                if seed_ids:
                    # 모든 아티스트별로 대표곡 3개와의 평균 유사도 계산
                    from collections import defaultdict
                    artist_songs_map = defaultdict(list)
                    for s in self.song_temps.values():
                        a = s.get('artist', '')
                        if a != seed_artist:
                            artist_songs_map[a].append(s)
                    
                    for other_artist, songs in artist_songs_map.items():
                        clean = other_artist.replace(' - Topic', '').strip().lower()
                        if clean in sim_map:
                            continue
                        # 대표곡 3개
                        top_songs = sorted(songs, key=lambda x: x.get('total_plays', 0), reverse=True)[:3]
                        target_ids = [s['song_id'] for s in top_songs]
                        
                        # 시드곡들 vs 대상곡들 교차 유사도 평균
                        pair_sims = []
                        for sid in seed_ids:
                            for tid in target_ids:
                                try:
                                    s = self.similarity_engine.calculate_similarity(sid, tid)
                                    if s is not None:
                                        pair_sims.append(s)
                                except:
                                    pass
                        if pair_sims:
                            sim_map[clean] = sum(pair_sims) / len(pair_sims)
            
            self._seed_artist_sim_cache[seed_artist] = sim_map

        sim_map = self._seed_artist_sim_cache[seed_artist]
        clean_target = target_artist.replace(' - Topic', '').strip().lower()
        return sim_map.get(clean_target, 0.0)

    def _load_artist_sim_from_cache(self, seed_artist):
        """미리 캐싱해둔 Last.fm 유사 아티스트 파일에서 직접 로드"""
        if not hasattr(self, '_lastfm_file_cache'):
            cache_path = os.path.join(
                os.path.dirname(__file__), '..', 'data', 'caches', 'lastfm_artist_sim_cache.json'
            )
            if os.path.exists(cache_path):
                import json
                with open(cache_path, 'r', encoding='utf-8') as f:
                    self._lastfm_file_cache = json.load(f)
                print(f"  [아티스트 유사도] 캐시 파일 로드: {len(self._lastfm_file_cache)}개 항목")
            else:
                self._lastfm_file_cache = {}
            
            # ★ 이름 별명 매핑 구축 (Last.fm 이름 → 라이브러리 이름)
            # 문제: Last.fm에서 JANNABI 유사로 '검정치마'를 반환하지만
            #        라이브러리에는 'The Black Skirts'로 저장됨
            # 해결: 캐시 키 이름과 라이브러리 이름의 교차점을 찾아서 별명 테이블 구축
            
            # 라이브러리 아티스트 이름 수집
            lib_names = set()
            for info in self.song_temps.values():
                a = info.get('artist', '').replace(' - Topic', '').strip()
                if a:
                    lib_names.add(a)
            
            # 캐시 키에서 아티스트 이름 추출
            cache_key_names = set()
            for key in self._lastfm_file_cache:
                if key.startswith('similar_artists||'):
                    cache_key_names.add(key.replace('similar_artists||', ''))
            
            # 동일 아티스트 = 캐시 키와 라이브러리 이름이 다르지만 같은 아티스트
            # 판별: 캐시 키 A의 유사 목록에서 라이브러리 이름 B가 나오고,
            #        캐시 키 B의 유사 목록에서 캐시 키 A가 나오면 → A = B
            self._alias_to_lib = {}  # lastfm이름(소문자) → 라이브러리이름(소문자)
            
            # 1단계: 직접 일치 (대소문자 무시)
            lib_lower = {n.lower(): n for n in lib_names}
            for n in lib_names:
                self._alias_to_lib[n.lower()] = n.lower()
            for ck in cache_key_names:
                if ck.lower() in lib_lower:
                    self._alias_to_lib[ck.lower()] = ck.lower()
            # 2단계: 수동 별명 테이블 (Last.fm 이름 → 라이브러리 이름)
            # Last.fm이 한국어/다른 표기로 반환하는 이름을 라이브러리 이름에 매핑
            ARTIST_ALIASES = {
                '검정치마': 'The Black Skirts',
                '혁오': 'HYUKOH',
                '언니네 이발관': 'Lim Kim',  # 관련 아티스트
                '브로콜리 너마저': 'Standing Egg',  # 관련
                '데이먼스 이어 damons year': 'Damons Year',
                '카더가든': 'Car, the Garden',
                '잔나비': 'JANNABI',
                '한로로': 'HANRORO',
                '너드커넥션': 'Nerd Connection',
                '적재': 'Jukjae',
                '장범준': 'Jang Beom June',
                '정승환': 'Jung Seung Hwan',
                '최유리': 'Choi Yu Ree',
                '실리카겔': 'Silica Gel',
            }
            for alias, lib_name in ARTIST_ALIASES.items():
                if lib_name.lower() in lib_lower:
                    self._alias_to_lib[alias.lower()] = lib_name.lower()
            
            mapped = sum(1 for v in self._alias_to_lib.values() if v in lib_lower)
            print(f"  [아티스트 매핑] {len(self._alias_to_lib)}개 별명 → {mapped}개 라이브러리 매핑")
        
        clean = seed_artist.replace(' - Topic', '').strip()
        cache_key = f"similar_artists||{clean}"
        entries = self._lastfm_file_cache.get(cache_key, [])
        
        sim_map = {}
        for entry in entries:
            lastfm_name = entry.get('name', '').lower()
            match = float(entry.get('match', 0))
            # Last.fm 이름으로 직접 등록
            sim_map[lastfm_name] = match
            # 별명으로도 등록 (예: '검정치마' → 'the black skirts')
            alias = self._alias_to_lib.get(lastfm_name)
            if alias and alias != lastfm_name:
                sim_map[alias] = match
        return sim_map

    def _get_artist_genre(self, artist):
        """아티스트의 대표 장르를 song_temps에서 추출 (최빈 장르)"""
        from collections import Counter
        genres = []
        for info in self.song_temps.values():
            if info.get('artist') == artist and info.get('genre'):
                genres.append(info['genre'])
        if genres:
            return Counter(genres).most_common(1)[0][0]
        return None

    def _genre_family(self, genre):
        """세부 장르를 대분류로 묶음 (K-Pop은 너무 광범위하므로 별도)"""
        if not genre:
            return None
        g = genre.lower()
        # K-Pop은 너무 넓어서 독립 분류 (아이돌/밴드/발라드 다 섞임)
        if g in ['k-pop', 'kpop']:
            return 'k-pop'
        elif any(k in g for k in ['indie rock', 'indie', 'alternative']):
            return 'indie'
        elif any(k in g for k in ['rock', 'punk', 'metal']):
            return 'rock'
        elif any(k in g for k in ['ballad', 'adult contemporary']):
            return 'ballad'
        elif any(k in g for k in ['r&b', 'soul', 'r&b/soul']):
            return 'rnb'
        elif any(k in g for k in ['pop']):
            return 'pop'
        elif any(k in g for k in ['hip-hop', 'rap', 'hip hop']):
            return 'hiphop'
        elif any(k in g for k in ['electronic', 'edm', 'house', 'techno']):
            return 'electronic'
        elif any(k in g for k in ['singer/songwriter', 'folk']):
            return 'singer-songwriter'
        return 'other'

    def _calculate_similarity(self, target_song, seed_song=None, seed_vector=None):
        """
        [맥락 점수: 곡 유사도 60% + 아티스트 유사도 40%]
        두 축을 합산하여 0~100% 범위로 반환합니다.

        곡 유사도: Multi-Signal (태그 25% + 가사 3% + 오디오 58% + 메타 14%)
        아티스트 유사도: Last.fm similar artists 매칭 점수
        """
        W_SONG = 0.6
        W_ARTIST = 0.4

        # --- 곡 유사도 (0~1) --- Multi-Signal 우선, seed_vector는 fallback
        song_sim = 0.0
        if seed_song and hasattr(self, 'similarity_engine') and self.similarity_engine:
            # 전체 4신호 사용 (학습된 가중치 적용)
            target_id = target_song.get('song_id')
            seed_id = seed_song.get('song_id')
            if target_id and seed_id:
                song_sim = float(self.similarity_engine.calculate_similarity(target_id, seed_id))
        elif seed_vector is not None and hasattr(self, 'similarity_engine') and self.similarity_engine:
            # fallback: 태그 TF-IDF 벡터 기반
            target_id = target_song.get('song_id')
            if target_id:
                song_sim = float(self.similarity_engine.calculate_similarity_to_vector(target_id, seed_vector))

        # --- 아티스트 유사도 (0~1) ---
        artist_sim = 0.0
        seed_artist = None
        if seed_song:
            seed_artist = seed_song.get('artist', '')
        target_artist = target_song.get('artist', '')
        if seed_artist and target_artist:
            artist_sim = self._get_artist_similarity(target_artist, seed_artist)

        # --- 합산 ---
        context = song_sim * W_SONG + artist_sim * W_ARTIST
        return context * 100.0

    def generate_playlist(self, total_songs=20, preset='default', custom_ratios=None, seed_tracks=None):
        """
        [v2] 연속 점수 기반으로 플레이리스트를 생성한다.

        최종 가중치 = affinity^w_a × momentum^w_m × context^w_c + noise
        preset='lgbm'일 때: lgbm_score^w × context^w + noise
        이산적 라벨 없이, 점수 순으로 곡을 선택한다.
        """
        use_lgbm = (preset == 'lgbm')
        preset_config = self.PRESETS.get(preset, self.PRESETS['default'])
        w_c = preset_config.get('w_context', 1.0)
        discovery_ratio = preset_config['discovery_ratio']

        if use_lgbm:
            w_lgbm = preset_config.get('w_lgbm', 1.0)
            print(f"  🤖 LightGBM 모드: predicted_affinity 기반 (w_lgbm={w_lgbm}, w_context={w_c})")
        else:
            w_a = preset_config['w_affinity']
            w_m = preset_config['w_momentum']

        n_discovery = int(round(total_songs * discovery_ratio))
        n_main = total_songs - n_discovery

        playlist = []
        seed_song = None
        seed_vector = None

        # 아티스트 유사도 캐시 초기화 (플레이리스트 생성마다 리셋)
        self._seed_artist_sim_cache = {}

        if seed_tracks and hasattr(self, 'similarity_engine') and self.similarity_engine:
            seed_vector = self.similarity_engine.build_seed_vector(seed_tracks)
            # Multi-Signal용: 시드 트랙에 해당하는 곡을 라이브러리에서 찾기
            for st in seed_tracks:
                st_title = st.get('title', '').lower()
                st_artist = st.get('artist', '').lower()
                for info in self.song_temps.values():
                    lib_title = info.get('title', '').lower()
                    lib_artist = info.get('artist', '').lower()
                    # 정확 일치 또는 부분 포함 매칭
                    title_match = (lib_title == st_title or
                                   st_title in lib_title or
                                   lib_title in st_title)
                    artist_match = (lib_artist == st_artist or
                                    st_artist in lib_artist)
                    if title_match and artist_match:
                        seed_song = info
                        print(f"  🎯 시드 매칭: {info.get('title', '?')[:40]}")
                        break
                if seed_song:
                    break

        # 시드가 없으면 상위 20곡 중 랜덤 선택
        if seed_vector is None:
            import random
            all_songs = list(self.song_temps.values())
            if all_songs:
                if use_lgbm and self.lgbm_scores:
                    scored = sorted(all_songs,
                        key=lambda x: self.lgbm_scores.get(x.get('song_id', ''), 0),
                        reverse=True)
                else:
                    scored = sorted(all_songs,
                        key=lambda x: x.get('affinity', 0) * x.get('momentum', 0),
                        reverse=True)
                top_pool = scored[:min(20, len(scored))]
                seed_song = random.choice(top_pool)
                print(f"  🎯 자동 시드: {seed_song.get('title', '?')[:40]} "
                      f"({seed_song.get('artist', '?').replace(' - Topic', '')})")

        # 다양성 제한
        max_per_artist = max(2, int(total_songs * 0.2))
        artist_counts = defaultdict(int)

        # === Main 슬롯: 가중치 정렬 ===
        scored_pool = []
        for song_id, info in self.song_temps.items():
            # 맥락 점수 (시드 있을 때만)
            context_score = 1.0
            if seed_vector is not None or seed_song:
                sim = self._calculate_similarity(info, seed_song=seed_song, seed_vector=seed_vector)
                context_score = max(0.01, sim / 100.0)  # 0~1 범위로

            if use_lgbm:
                # LightGBM 모드: 호감도 45% + 유사도 55% 가중합 (학습 결과)
                lgbm_score = self.lgbm_scores.get(song_id, 2.5) / 5.0  # 0~1 범위로 정규화
                final_weight = (
                    w_lgbm * lgbm_score + w_c * context_score
                    + np.random.uniform(0, 0.02)
                )
            else:
                # 기존 모드: affinity x momentum x context
                affinity = info.get('affinity', 0.5)
                momentum = info.get('momentum', 0.5)
                final_weight = (
                    (affinity ** w_a) * (momentum ** w_m) * (context_score ** w_c)
                    + np.random.uniform(0, 0.05)
                )

            song_copy = info.copy()
            song_copy['similarity_score'] = round(context_score * 100, 1)
            song_copy['final_weight'] = round(final_weight, 4)
            if use_lgbm:
                song_copy['lgbm_score'] = round(self.lgbm_scores.get(song_id, 2.5), 2)
            scored_pool.append(song_copy)

        # 가중치 순 정렬
        scored_pool.sort(key=lambda x: x['final_weight'], reverse=True)

        # === Discovery 슬롯 (우선 선택) ===
        chosen_discovery = []
        if n_discovery > 0:
            discovery_songs = self._get_discovery_songs(
                n_discovery, seed_song=seed_song, seed_vector=seed_vector,
                preset=preset, seed_tracks=seed_tracks
            )
            for song in discovery_songs:
                if artist_counts[song['artist']] < max_per_artist:
                    chosen_discovery.append(song)
                    artist_counts[song['artist']] += 1
                if len(chosen_discovery) >= n_discovery:
                    break

        # 부족한 Discovery 슬롯은 Main으로 채움
        n_main_actual = total_songs - len(chosen_discovery)

        # === Main 곡 우선 선택 (아티스트 캡 적용) ===
        main_songs = []
        for song in scored_pool:
            if artist_counts[song['artist']] < max_per_artist:
                song['reason'] = 'Main'
                main_songs.append(song)
                artist_counts[song['artist']] += 1
            if len(main_songs) >= n_main_actual:
                break

        playlist.extend(main_songs)
        playlist.extend(chosen_discovery)

        # DJ 스타일 곡 배치
        playlist = self._arrange_playlist_order(playlist)

        return playlist

    def _arrange_playlist_order(self, playlist):
        """
        [DJ 스타일 곡 배치 v2 — 가중치 티어 기반 자연스러운 인터리브]
        """
        known_songs = []
        discovery_songs = []

        for s in playlist:
            reason = s.get('reason', s.get('temperature', ''))
            if 'Discovery' in reason:
                discovery_songs.append(s)
            else:
                known_songs.append(s)

        # 가중치(final_weight) 기준으로 정렬 후 3개 티어로 분할
        known_songs.sort(key=lambda x: x.get('final_weight', 0), reverse=True)
        n_known = len(known_songs)
        t1 = max(1, n_known // 3)
        t2 = max(2, (n_known * 2) // 3)

        buckets = {
            'High': known_songs[:t1],
            'Mid': known_songs[t1:t2],
            'Low': known_songs[t2:]
        }

        # 버킷 내에서는 랜덤 셔플 (매번 다른 순서)
        for b in buckets.values():
            if len(b) > 0:
                np.random.shuffle(b)

        # 티어별 라운드로빈 인터리브 (고 -> 저 -> 중 흐름)
        interleave_order = ['High', 'Low', 'Mid']
        interleaved = []
        
        # 첫 곡은 가장 가중치가 높은 그룹에서
        if buckets['High']:
            interleaved.append(buckets['High'].pop(0))
        elif buckets['Mid']:
            interleaved.append(buckets['Mid'].pop(0))

        round_idx = 0
        while len(interleaved) < len(known_songs):
            tier = interleave_order[round_idx % len(interleave_order)]
            if buckets[tier]:
                interleaved.append(buckets[tier].pop(0))
            round_idx += 1
            if round_idx > len(known_songs) * 3:
                for b in buckets.values():
                    interleaved.extend(b)
                    b.clear()
                break

        # Discovery 슬롯 삽입 (4곡 간격, 최소 4번째부터)
        discovery_songs.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)

        total = len(playlist)
        result = []
        known_idx = 0
        disc_idx = 0

        disc_interval = max(3, (total - 1) // (len(discovery_songs) + 1)) if discovery_songs else total
        disc_slots = set()
        if discovery_songs:
            first_slot = min(4, total - 1)
            for i in range(len(discovery_songs)):
                slot = first_slot + i * disc_interval
                if slot < total:
                    disc_slots.add(slot)

        for i in range(total):
            if i in disc_slots and disc_idx < len(discovery_songs):
                result.append(discovery_songs[disc_idx])
                disc_idx += 1
            elif known_idx < len(interleaved):
                result.append(interleaved[known_idx])
                known_idx += 1

        # 남은 곡 추가
        result.extend(interleaved[known_idx:])
        result.extend(discovery_songs[disc_idx:])

        result = result[:total]

        # === 아티스트 연속 방지 패스 ===
        # 같은 아티스트가 2곡 연속 나오면, 뒤쪽에서 다른 아티스트 곡과 교환
        for _ in range(3):  # 최대 3회 반복 (수렴용)
            swapped = False
            for i in range(1, len(result)):
                if result[i].get('artist') == result[i-1].get('artist'):
                    # i번 곡을 뒤쪽의 다른 아티스트 곡과 교환
                    for j in range(i+1, len(result)):
                        if (result[j].get('artist') != result[i].get('artist') and
                            (j == len(result)-1 or result[j].get('artist') != result[j+1].get('artist', '')) and
                            (j == 0 or result[j].get('artist') != result[j-1].get('artist', ''))):
                            result[i], result[j] = result[j], result[i]
                            swapped = True
                            break
            if not swapped:
                break

        return result

    def _get_discovery_songs(self, n, seed_song=None, seed_vector=None, preset='default', seed_tracks=None):
        """
        [통합 Discovery 엔진]
        내부 Discovery(기존 라이브러리의 덜 들은 곡)와
        외부 Discovery(Last.fm 기반 진짜 새 곡)를 비율에 맞게 혼합.

        내부 소스:
          - 성장 신호 아티스트의 덜 들은 곡
          - Discovery_Candidate (아티스트 생존율 높은 미탐색 곡)
        외부 소스:
          - ExternalDiscoveryEngine (Last.fm 유사 아티스트 → 인기곡 → 태그 필터)
        """
        # === 내부/외부 할당 수량 계산 ===
        mix = self.DISCOVERY_MIX.get(preset, self.DISCOVERY_MIX['default'])
        n_internal = max(1, round(n * mix['internal']))
        n_external = max(1, n - n_internal)

        # === [내부 Discovery] ===
        internal = []

        # 1순위: 성장 신호 아티스트의 적게 들은 곡
        for sig in self.growth_signals:
            artist = sig['artist']
            trigger_id = sig['trigger_song_id']

            artist_songs = [
                s for s in self.song_temps.values()
                if s['artist'] == artist
                and s['song_id'] != trigger_id
                and s['total_plays'] < 5
            ]

            for song in artist_songs[:2]:
                song_copy = song.copy()
                song_copy['reason'] = f'Discovery·내부 (← {sig["trigger_song"][:20]})'
                song_copy['similarity_score'] = self._calculate_similarity(song, seed_song=seed_song, seed_vector=seed_vector)
                song_copy['discovery_source'] = 'internal'
                internal.append(song_copy)

        # 2순위: Discovery_Candidate + 일반 덜 들은 곡 보충
        if len(internal) < n_internal:
            low_play_songs = [
                s for s in self.song_temps.values()
                if s['total_plays'] <= 3
                and s.get('affinity', 0) >= 0.3  # v2: 호감도 기반 필터 (temperature 라벨 대신)
            ]
            
            scored_low = []
            for song in low_play_songs:
                sim = self._calculate_similarity(song, seed_song=seed_song, seed_vector=seed_vector)
                if sim < 15:
                    continue
                weight = sim + np.random.uniform(0, 0.5)
                scored_low.append((song, weight))
            
            scored_low.sort(key=lambda x: x[1], reverse=True)
            
            for song, w in scored_low[:n_internal - len(internal)]:
                song_copy = song.copy()
                song_copy['reason'] = 'Discovery·내부 (새 발견)'
                song_copy['similarity_score'] = round(self._calculate_similarity(song, seed_song=seed_song, seed_vector=seed_vector), 1)
                song_copy['discovery_source'] = 'internal'
                internal.append(song_copy)

        # 내부 아티스트 중복 제거 (아티스트당 1곡)
        internal_deduped = []
        seen_artists = set()
        for d in internal:
            a = d.get('artist', '').lower()
            if a not in seen_artists:
                internal_deduped.append(d)
                seen_artists.add(a)
        internal = internal_deduped[:n_internal]

        # === [외부 Discovery] ===
        external = []
        if self.external_engine and n_external > 0:
            try:
                ext_recs = self.external_engine.discover_new_songs(
                    n=n_external,
                    discovery_preset=preset,
                    seed_tracks=seed_tracks
                )
                for rec in ext_recs:
                    # 외부 곡을 내부 song_temps 형식에 맞게 변환
                    fam_icons = {'deep_dive': '🔵', 'expand': '🟢', 'explore': '🟡'}
                    fam_type = rec.get('fam_type', 'explore')
                    fam_count = rec.get('fam_count', 0)
                    fam_label = f"보유 {fam_count}곡" if fam_count > 0 else "신규 아티스트"
                    
                    external.append({
                        'song_id': f"ext__{rec['artist']}__{rec['track']}",
                        'title': rec['track'],
                        'artist': rec['artist'],
                        'tier': 'NEW',
                        'temperature': 'Discovery',
                        'total_plays': 0,
                        'skip_rate': 0,
                        'days_since_last': 0,
                        'recent_30d_plays': 0,
                        'recent_14d_plays': 0,
                        'trend_slope': 0,
                        'peak_time_of_day': 'Unknown',
                        'genre': None,
                        'reason': f'Discovery·외부 {fam_icons.get(fam_type, "")} ({fam_label}, ← {rec.get("source_artist", "?")[:15]})',
                        'similarity_score': round(rec.get('tag_similarity', 0) * 100, 1),
                        'discovery_source': 'external',
                        'fam_type': fam_type,
                    })
            except Exception as e:
                print(f"  [외부 Discovery 경고] {e}")

        # === 내부 + 외부 병합 ===
        # 내부를 먼저 배치하고, 외부를 뒤에 채움 (DJ 배치에서 간격 조절됨)
        combined = internal + external

        # 부족한 경우 상대편에서 보충
        if len(combined) < n:
            # 외부가 부족하면 내부에서 더 채움 (또는 그 반대)
            pass  # 이미 각자 채울 수 있는 만큼 채웠으므로 부족분은 그대로 둠

        return combined[:n]

    def display_playlist(self, playlist, preset_name='default'):
        """생성된 플레이리스트를 보기 좋게 출력한다."""
        preset_names_kr = {
            'default': '🎵 기본 모드',
            'explore': '🔍 탐험 모드',
            'comfort': '🏠 편안함 모드',
            'nostalgia': '🕰️ 추억 모드'
        }

        print("\n" + "=" * 60)
        print(f"🎶 추천 플레이리스트 — {preset_names_kr.get(preset_name, preset_name)}")
        print("=" * 60)

        for i, song in enumerate(playlist, 1):
            if song.get('discovery_source') == 'external':
                icon = '🆕'
            else:
                m = song.get('momentum', 0.5)
                a = song.get('affinity', 0.5)
                if m > 0.8:
                    icon = '🔥'
                elif a > 0.8:
                    icon = '💖'
                elif m < 0.2:
                    icon = '🧊'
                else:
                    icon = '🎵'
                    
            reason = song.get('reason', 'Main')
            title_short = song['title'][:33]
            artist_short = song['artist'][:18]
            sim_score = song.get('similarity_score', 0)
            
            print(f"  {i:>2}. {icon} {title_short:<33} — {artist_short}")
            if song.get('discovery_source') == 'external':
                print(f"       [{reason}] 🆕 처음 듣는 곡 | 태그 일치율: {sim_score:>5.1f}%")
            elif hasattr(self, 'similarity_engine') and self.similarity_engine:
                print(f"       [{reason}] 재생 {song.get('total_plays',0):>2}회 | 태그 일치율: {sim_score:>5.1f}%")
            else:
                print(f"       [{reason}] 재생 {song.get('total_plays',0):>2}회 | 장르/아티스트 매칭: {sim_score:>4.1f}점")

        # 비율 통계
        print("\n--- 구성 비율 ---")
        reason_counts = defaultdict(int)
        disc_internal = 0
        disc_external = 0
        for s in playlist:
            r = s.get('reason', 'Unknown')
            if 'Discovery' in r:
                reason_counts['Discovery'] += 1
                if s.get('discovery_source') == 'external':
                    disc_external += 1
                else:
                    disc_internal += 1
            else:
                reason_counts[r] += 1
        for r, c in sorted(reason_counts.items()):
            pct = c / len(playlist) * 100
            if r == 'Discovery' and (disc_internal + disc_external) > 0:
                print(f"  {r}: {c}곡 ({pct:.0f}%) — 내부 {disc_internal}곡 / 외부 🆕 {disc_external}곡")
            else:
                print(f"  {r}: {c}곡 ({pct:.0f}%)")


# =============================================================================
# 메인 실행: 전체 파이프라인
# =============================================================================
def run_pipeline(csv_path, user_name, playlist_size=20, preset='default', metadata_path=None, user_birth_year=1998, seed_tracks=None, lastfm_api_key=None, skip_external=False):
    """전체 파이프라인을 실행한다. (내부 + 외부 Discovery 통합)"""
    print(f"\n{'#' * 60}")
    print(f"# 🎧 {user_name} — 음원 생애주기 기반 추천 시스템")
    print(f"{'#' * 60}")

    LASTFM_API_KEY = lastfm_api_key or os.environ.get("LASTFM_API_KEY", "")

    # 데이터 로드
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"\n📁 데이터 로드 완료: {len(df)}건, 곡 {df['song_id'].nunique()}개, "
          f"아티스트 {df['artist'].nunique()}명 (정규화 전)")

    # Step 0: 아티스트명 정규화
    print("\n[Step 0/6] 아티스트명 정규화...")
    normalizer = ArtistNameNormalizer()
    df = normalizer.normalize_dataframe(df)
    normalizer.summary()
    print(f"  → 정규화 후 아티스트: {df['artist'].nunique()}명")

    # Step 1: 아티스트 Tier 분류
    print("\n[Step 1/6] 아티스트 Tier 클러스터링...")
    tier_classifier = ArtistTierClassifier(df)
    tier_map = tier_classifier.classify_tiers()
    tier_classifier.summary()

    # Step 2: 곡별 연속 점수 산출 (v2)
    print("\n[Step 2/6] 곡별 점수(Affinity/Momentum) 산출...")
    scorer = SongScorer(df, tier_map, metadata_path=metadata_path, user_birth_year=user_birth_year)
    song_scores = scorer.score_all_songs()
    scorer.summary()

    # Step 3: 비대칭 흐름 감지 (참고용, 점수에는 이미 모멘텀으로 반영됨)
    print("\n[Step 3/6] 비대칭 흐름(성장/쇠퇴) 감지...")
    flow_detector = AsymmetricFlowDetector(df, tier_map)
    growth = flow_detector.detect_growth_bottom_up()
    decline = flow_detector.detect_decline_top_down()
    flow_detector.summary()

    # Step 4: Multi-Signal 기반 유사도 엔진 준비
    similarity_engine = None
    try:
        from tag_similarity import TagSimilarityEngine
        from multi_signal_engine import MultiSignalSimilarityEngine
        
        tag_engine = None
        lyrics_engine = None
        audio_engine = None
        
        if LASTFM_API_KEY and not skip_external:
            tag_engine = TagSimilarityEngine(api_key=LASTFM_API_KEY)
            # v2: momentum > 0.05인 곡만 태그 매칭
            active_songs = {k: v for k, v in song_scores.items() if v.get('momentum', 0) > 0.05}
            tag_engine.build_tag_vectors(active_songs)
            
        try:
            from lyrics_engine import LyricsEngine
            from audio_features_engine import AudioFeaturesEngine
            lyrics_engine = LyricsEngine(genius_token=os.environ.get("GENIUS_TOKEN", ""))
            audio_engine = AudioFeaturesEngine()
            audio_engine._build_matrix()
        except Exception as e:
            print(f"  [주의] Lyrics/Audio 엔진 로드 실패: {e}")
            
        similarity_engine = MultiSignalSimilarityEngine(
            tag_engine=tag_engine,
            lyrics_engine=lyrics_engine,
            audio_engine=audio_engine,
            metadata_path=metadata_path
        )
        print("\n[Step 4/6] Multi-Signal 유사도 엔진 로드 완료 ✅ (Tags + Lyrics + Audio + Meta)")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"\n[주의] 유사도 엔진 초기화 실패, 기본 로직 사용 ({e})")

    # Step 5: 외부 신곡 발굴 엔진 준비
    external_engine = None
    if not skip_external:
        try:
            from external_discovery import ExternalDiscoveryEngine
            if LASTFM_API_KEY:
                external_engine = ExternalDiscoveryEngine(
                    api_key=LASTFM_API_KEY,
                    song_temps=song_scores,
                    similarity_engine=similarity_engine
                )
                print("\n[Step 5/6] 외부 신곡 발굴 엔진(Last.fm) 준비 완료 ✅")
        except Exception as e:
            print(f"\n[주의] 외부 Discovery 엔진 초기화 실패 ({e})")
    else:
        print("\n[Step 5/6] 외부 Discovery 건너뜀 (skip_external=True)")

    # Step 6: 플레이리스트 생성 (내부 + 외부 Discovery 통합)
    print(f"\n[Step 6/6] 추천 플레이리스트 생성 (프리셋: {preset})...")

    # LightGBM 점수 로드 (preset='lgbm'일 때)
    lgbm_scores = None
    if preset == 'lgbm':
        lgbm_csv = os.path.join(
            os.path.dirname(csv_path), f'{user_name}_lgbm_v4_ranking.csv'
        )
        if os.path.exists(lgbm_csv):
            lgbm_df = pd.read_csv(lgbm_csv, encoding='utf-8-sig')
            # title+artist -> song_id 매핑
            sid_map = {}
            for sid, info in song_scores.items():
                key = (info.get('title', ''), info.get('artist', ''))
                sid_map[key] = sid
            lgbm_scores = {}
            for _, row in lgbm_df.iterrows():
                key = (row['title'], row['artist'])
                sid = sid_map.get(key)
                if sid and pd.notna(row.get('predicted_affinity')):
                    lgbm_scores[sid] = float(row['predicted_affinity'])
            print(f"  🤖 LightGBM v4 랭킹 로드: {len(lgbm_scores)}곡")
        else:
            print(f"  ⚠️ LightGBM 랭킹 파일 없음: {lgbm_csv}, default로 fallback")
            preset = 'default'

    mixer = PlaylistMixer(
        song_scores, growth,
        similarity_engine=similarity_engine,
        external_engine=external_engine,
        lgbm_scores=lgbm_scores
    )
    playlist = mixer.generate_playlist(total_songs=playlist_size, preset=preset, seed_tracks=seed_tracks)
    mixer.display_playlist(playlist, preset)

    return {
        'tier_classifier': tier_classifier,
        'scorer': scorer,
        'flow_detector': flow_detector,
        'mixer': mixer,
        'playlist': playlist,
        'external_engine': external_engine
    }


# =============================================================================
# 실행
# =============================================================================
if __name__ == '__main__':
    base = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\유튜브 뮤직 로그들'

    # 친구D 데이터로 실행
    result_seo = run_pipeline(
        csv_path=f'{base}\\친구D\\친구D_features.csv',
        user_name='친구D',
        playlist_size=20,
        preset='default',
        metadata_path=r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\ytm_metadata_cache.csv',
        user_birth_year=1998
    )

    # 친구B 데이터로 실행
    result_kim = run_pipeline(
        csv_path=f'{base}\\친구B\\친구B_features.csv',
        user_name='친구B',
        playlist_size=20,
        preset='default',
        metadata_path=r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\ytm_metadata_cache.csv',
        user_birth_year=1998
    )
