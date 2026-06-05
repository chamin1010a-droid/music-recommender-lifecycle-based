"""파이프라인 실행 → song_scores를 JSON으로 내보내기"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from lifecycle_recommender import (
    ArtistNameNormalizer, ArtistTierClassifier, SongScorer
)
import pandas as pd

CSV_PATH = r'유튜브 뮤직 로그들\user\user_features.csv'
META_PATH = r'data\caches\ytm_metadata_cache.csv'

print("파이프라인 실행 중...")
df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')

# Step 0: 아티스트명 정규화
normalizer = ArtistNameNormalizer()
df = normalizer.normalize_dataframe(df)

# Step 1: Tier 분류
tier_classifier = ArtistTierClassifier(df)
tier_map = tier_classifier.classify_tiers()

# Step 2: 곡 점수
scorer = SongScorer(df, tier_map, metadata_path=META_PATH)
song_scores = scorer.score_all_songs()

# JSON 내보내기
output = {}
for song_id, scores in song_scores.items():
    output[song_id] = {
        'affinity': scores.get('affinity', 0),
        'momentum': scores.get('momentum', 0),
        'zone_label': scores.get('zone_label', ''),
        'total_plays': scores.get('total_plays', 0),
        'skip_rate': scores.get('skip_rate', 0),
        'genre': scores.get('genre', ''),
    }

out_path = r'data\caches\song_scores.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n완료: {len(output)}곡 점수 -> {out_path}")
print(f"  평균 affinity: {sum(v['affinity'] for v in output.values())/len(output):.3f}")
print(f"  평균 momentum: {sum(v['momentum'] for v in output.values())/len(output):.3f}")
