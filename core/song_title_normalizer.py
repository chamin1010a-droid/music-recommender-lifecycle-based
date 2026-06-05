"""
[곡 제목 별칭 정규화 시스템]
한국 곡들의 영어↔한글 제목을 매핑하여 어느 이름으로든 검색 가능하게 합니다.

3단계 자동 매핑:
  1. 괄호 안 한글 추출: "Blue spring (작전명 청-춘!)" → "작전명 청-춘!"
  2. iTunes 한글 제목: iTunes에 한글 제목이 있으면 매핑
  3. 수동 매핑: 자동으로 잡을 수 없는 유명곡 수동 등록
"""

import os
import json
import re
import pandas as pd


class SongTitleNormalizer:
    """
    곡 제목의 한글↔영문 별칭을 관리합니다.
    song_id로 검색하거나, 한글/영문 제목으로 song_id를 역검색할 수 있습니다.
    """
    
    # 수동 매핑: {아티스트(정제): {영어제목: 한글별칭}}
    MANUAL_ALIASES = {
        'the black skirts': {
            # 공식 한국어 제목이 확인된 곡만
            'Holiday': '빨간 나를',
            'Antifreeze': '부동액',
            'My Little Lambs': '어린양',
            'My shadow': '그늘은 그림자로',
        },
        'jannabi': {
            'for lovers who hesitate': '주저하는 연인들을 위해',
            'Summer': '뜨거운 여름밤은 가고 남은 건 볼품없지만',
            'A thought on an autumn night': '가을밤에 든 생각',
            'dreams, books, power and walls': '꿈과 책과 힘과 벽',
            'Good Luck to You': '행운을 빌어요',
            'Tell me': '꿈나라 별나라',
            'for lovers who hesitate': '주저하는 연인들을 위해',
            'Baby I need you': '사랑하긴 했었나요',
            'See Your Eyes': '너의 눈을 봐',
            'November Rain': '11월의 비',
            'Step': '한걸음',
            'Good Good Night': '나의 기쁨 나의 노래',
            'Sweet memories': '그 밤 그 밤',
            'Blue spring': '작전명 청-춘!',
            'bad dreams': '나쁜 꿈',
            'The Land of Fantasy': '환상의 나라',
            'land of night': '새 어둠 새 눈',
            'The King of Romance': '로맨스의 왕',
            'Confession Show': '고백극장',
            'Come Back Home': '컴백홈',
            'mirror': '거울',
            'about a boy': '우리 애는요',
            'LEGEND': '전설',
            'TOGETHER!': '투게더!',
            'pony': '포니',
            'Like when we first met': '처음 만났을 때처럼',
            'LADYBIRD': '레이디버드',
            'Beautiful': '너 같아',
            'Goodbye Dreamin\' Old Stars': '굿바이 환상의 나라',
            'Pole Dance': '봉춤을 추네',
            'Winter is Coming': '누구나 겨울이 오면',
            'Clay Pigeon Boy': '소년 클레이 피전',
            'Sunshine comedy club': '선샤인코메디클럽',
            'Good Boy Twist': '굿보이 트위스트',
            'Old dog': '늙은 개',
            'Time': '누구를 위한 노래였던가',
            'waltz': '왕눈이 왈츠',
            'ASTEARSGOBY': '슬픔이여안녕',
        },
        'hanroro': {
            'To. __': '나에게',
            'Light or Rain': '빛 혹은 비',
            'H O M E': '홈',
        },
        'car, the garden': {
            'Romantic Sunday': '로맨틱 선데이',
            'Dreamed a dream': '꿈을 꿨어',
            'TEAM 401': '팀 401',
        },
        'day6': {
            'Shoot Me': '슛미',
            'Congratulations': '콩그레츄레이션스',
            'You Were Beautiful': '예뻤어',
            'I Loved You': '아이 러브드 유',
        },
        'hyukoh': {
            'TOMBOY': '톰보이',
            'Leather Jacket': '가죽자켓',
        },
        'nerd connection': {
            'Silently Completely Eternally': '조용히 완전히 영원히',
        },
    }
    
    def __init__(self, df=None, metadata_path=None, alias_cache_path=None):
        """
        df: 청취 기록 DataFrame (song_id, title, artist 필요)
        metadata_path: ytm_metadata_cache.csv 경로
        alias_cache_path: 별칭 매핑 캐시 저장 경로
        """
        self.alias_cache_path = alias_cache_path or os.path.join(
            os.path.dirname(__file__), '..', 'data', 'caches', 'song_title_aliases.json'
        )
        
        # song_id → {title_en, title_kr, aliases: []}
        self.song_aliases = {}
        
        # 역검색: 한글제목 → song_id, 영문제목 → song_id
        self._reverse_map = {}
        
        if os.path.exists(self.alias_cache_path):
            self._load_cache()
        
        if df is not None:
            self.build_aliases(df, metadata_path)
    
    def _load_cache(self):
        try:
            with open(self.alias_cache_path, 'r', encoding='utf-8') as f:
                self.song_aliases = json.load(f)
            self._build_reverse_map()
            print(f"  [곡 제목] 별칭 캐시 로드: {len(self.song_aliases)}곡")
        except:
            pass
    
    def _save_cache(self):
        os.makedirs(os.path.dirname(self.alias_cache_path), exist_ok=True)
        with open(self.alias_cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.song_aliases, f, ensure_ascii=False, indent=1)
    
    def _build_reverse_map(self):
        """역검색 맵 구축"""
        self._reverse_map = {}
        for sid, info in self.song_aliases.items():
            # 모든 별칭을 소문자로 역매핑
            for alias in info.get('aliases', []):
                key = alias.lower().strip()
                if key:
                    self._reverse_map[key] = sid
            # 원래 제목도
            if info.get('title_en'):
                self._reverse_map[info['title_en'].lower().strip()] = sid
            if info.get('title_kr'):
                self._reverse_map[info['title_kr'].lower().strip()] = sid
    
    def build_aliases(self, df, metadata_path=None):
        """3단계 자동 매핑 수행"""
        unique = df[['song_id', 'title', 'artist']].drop_duplicates('song_id')
        
        # iTunes 메타 로드
        itunes_map = {}
        if metadata_path and os.path.exists(metadata_path):
            meta = pd.read_csv(metadata_path, encoding='utf-8-sig')
            for _, r in meta.iterrows():
                sid = r.get('song_id')
                it = r.get('itunes_title')
                if sid and pd.notna(it):
                    itunes_map[sid] = str(it)
        
        count_auto = 0
        count_manual = 0
        
        for _, row in unique.iterrows():
            sid = row['song_id']
            title = str(row['title'])
            artist = str(row['artist']).replace(' - Topic', '').strip()
            artist_lower = artist.lower()
            
            aliases = set()
            title_kr = None
            title_en = title  # 기본은 원래 제목
            
            # === 1단계: 괄호 안 한글 추출 ===
            paren_match = re.findall(r'\(([^)]+)\)', title)
            for pm in paren_match:
                if any('\uac00' <= c <= '\ud7a3' for c in pm):
                    title_kr = pm.strip()
                    aliases.add(title_kr)
                    # 괄호 밖 부분이 영문이면 그것도 별칭
                    eng_part = re.sub(r'\s*\([^)]*\)', '', title).strip()
                    if eng_part and eng_part != title:
                        aliases.add(eng_part)
            
            # 제목 자체가 한글이면
            if not title_kr and any('\uac00' <= c <= '\ud7a3' for c in title):
                title_kr = title
            
            # === 2단계: iTunes 한글 제목 ===
            itunes_title = itunes_map.get(sid)
            if itunes_title and itunes_title != title:
                aliases.add(itunes_title)
                if not title_kr and any('\uac00' <= c <= '\ud7a3' for c in itunes_title):
                    title_kr = itunes_title
            
            # === 3단계: 수동 매핑 ===
            manual = self.MANUAL_ALIASES.get(artist_lower, {})
            # 제목의 괄호 제거 후 매칭
            clean_title = re.sub(r'\s*\([^)]*\)', '', title).strip()
            
            manual_kr = manual.get(title) or manual.get(clean_title)
            if manual_kr:
                title_kr = manual_kr
                aliases.add(manual_kr)
                count_manual += 1
            
            if aliases:
                count_auto += 1
            
            self.song_aliases[sid] = {
                'title_en': title_en,
                'title_kr': title_kr,
                'artist': artist,
                'aliases': list(aliases),
            }
        
        self._build_reverse_map()
        self._save_cache()
        
        kr_count = sum(1 for v in self.song_aliases.values() if v.get('title_kr'))
        print(f"  [곡 제목] 별칭 매핑 완료: 전체 {len(self.song_aliases)}곡")
        print(f"    한글 제목 보유: {kr_count}곡")
        print(f"    수동 매핑 적용: {count_manual}곡")
    
    def find_song(self, query, artist=None):
        """
        한글 또는 영문 제목으로 song_id를 찾습니다.
        부분 매칭도 지원합니다.
        artist: 아티스트명으로 필터 (부분 매칭)
        Returns: [(song_id, title_display, score), ...]
        """
        query_lower = query.lower().strip()
        artist_lower = artist.lower().strip() if artist else None
        
        # 한글 아티스트 → 영문명 매핑
        ARTIST_KR_MAP = {
            '검정치마': 'the black skirts', '잔나비': 'jannabi',
            '한로로': 'hanroro', '카더가든': 'car, the garden',
            '혁오': 'hyukoh', '널': 'nell', '버즈': 'buzz',
            '데이식스': 'day6', '아이유': 'iu', '방탄소년단': 'bts',
            '엑소': 'exo', '샤이니': 'shinee', '에스파': 'aespa',
            '엠씨더맥스': 'm.c the max', '먼데이키즈': 'monday kiz',
            '다비치': 'davichi', '쏜애플': 'thornapple',
            '볼빨간사춘기': 'bol4', '장범준': 'jang beom june',
        }
        
        def artist_matches(sid_artist):
            if not artist_lower:
                return True
            a = sid_artist.lower()
            query_a = ARTIST_KR_MAP.get(artist_lower, artist_lower)
            return query_a in a or a in query_a or artist_lower in a
        
        # 1. 정확 매칭
        if query_lower in self._reverse_map:
            sid = self._reverse_map[query_lower]
            info = self.song_aliases.get(sid, {})
            if artist_matches(info.get('artist', '')):
                display = self._display_title(sid)
                return [(sid, display, 1.0)]
        
        # 2. 부분 매칭
        results = []
        for sid, info in self.song_aliases.items():
            if not artist_matches(info.get('artist', '')):
                continue
            
            score = 0
            # 원래 제목에서 부분 매칭
            if query_lower in (info.get('title_en', '') or '').lower():
                score = max(score, 0.8)
            if info.get('title_kr') and query_lower in info['title_kr'].lower():
                score = max(score, 0.9)
            for alias in info.get('aliases', []):
                if query_lower in alias.lower():
                    score = max(score, 0.85)
            
            if score > 0:
                results.append((sid, self._display_title(sid), score))
        
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:10]
    
    def _display_title(self, sid):
        """표시용 곡명 생성: 한글 (영문)"""
        info = self.song_aliases.get(sid, {})
        en = info.get('title_en', '?')
        kr = info.get('title_kr')
        artist = info.get('artist', '')
        
        if kr and kr != en:
            return f"{kr} ({en}) — {artist}"
        return f"{en} — {artist}"
    
    def get_display_title(self, sid):
        """song_id → 표시용 제목 (한글 우선)"""
        info = self.song_aliases.get(sid, {})
        kr = info.get('title_kr')
        en = info.get('title_en', '?')
        return kr if kr else en
    
    def get_korean_title(self, sid):
        """song_id → 한글 제목 (없으면 None)"""
        info = self.song_aliases.get(sid, {})
        return info.get('title_kr')


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    csv_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
    meta_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\data\caches\ytm_metadata_cache.csv'
    
    df = pd.read_csv(csv_p, encoding='utf-8-sig')
    
    norm = SongTitleNormalizer(df, metadata_path=meta_p)
    
    # 테스트
    test_queries = [
        ('빨간 나를', None),
        ('나쁜 꿈', None),
        ('Holiday', None),
        ('Holiday', '검정치마'),
        ('주저하는 연인들을 위해', None),
        ('Love Is All', None),
        ('봉춤', None),
        ('가을밤', None),
        ('부동액', None),
        ('그림자', '검정치마'),
    ]
    
    print("\n=== 검색 테스트 ===")
    for item in test_queries:
        q = item[0]
        a = item[1] if len(item) > 1 else None
        label = f"'{q}'" + (f" (by {a})" if a else "")
        results = norm.find_song(q, artist=a)
        if results:
            sid, display, score = results[0]
            print(f"  {label:35s} → {display}")
        else:
            print(f"  {label:35s} → ❌ 못 찾음")
