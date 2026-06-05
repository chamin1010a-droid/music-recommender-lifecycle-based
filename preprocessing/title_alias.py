# title_alias.py
# 사용자가 한국어로 검색할 때 → 실제 데이터 속 영어 제목으로 매핑
# YouTube Music이 영어 제목으로 저장한 한국 곡들

TITLE_ALIAS = {
    # ==========================================
    # 검정치마 (The Black Skirts) - THIRSTY 앨범
    # ==========================================
    "빨간 나를": ("Holiday", "The Black Skirts - Topic"),
    "빨간나를": ("Holiday", "The Black Skirts - Topic"),
    "틀린질문": ("Wrong question", "The Black Skirts - Topic"),
    "섬": ("Island (queen of diamonds)", "The Black Skirts - Topic"),
    "상수역": ("Sangsu station", "The Black Skirts - Topic"),
    "광견일기": ("Mad dog diary", "The Black Skirts - Topic"),
    "하와이 검은 모래": ("Hawaiian black sand", "The Black Skirts - Topic"),
    "맑고 묽게": ("Thinner than water", "The Black Skirts - Topic"),
    "그늘은 그림자로": ("My shadow", "The Black Skirts - Topic"),
    "피와 갈증": ("Blood and thirst (king of hurts)", "The Black Skirts - Topic"),

    # 검정치마 - TEEN TROUBLES 앨범
    "불세례": ("Baptized In Fire", "The Black Skirts - Topic"),
    "어린양": ("My Little Lambs", "The Black Skirts - Topic"),
    "매미들": ("Cicadas", "The Black Skirts - Topic"),
    "따라갈래": ("Follow You", "The Black Skirts - Topic"),
    "미는 남자": ("Min", "The Black Skirts - Topic"),

    # ==========================================
    # 잔나비 (JANNABI) - 이미 괄호 안에 한글 병기되어 있어서 
    # 데이터에 "See Your Eyes (See Your Eyes)" 형태로 저장됨
    # 따로 alias 불필요 (이미 title에 한글 포함)
    # ==========================================

    # ==========================================
    # M.C the MAX - 일부 영어/한글 혼재
    # 데이터 확인 결과 대부분 영문으로 저장
    # ==========================================
    "하루살이": ("One Day Only (사계) (하루살이)", "M.C the MAX - Topic"),
    "입술의 말": ("Lying On Your Lips (입술의 말)", "M.C the MAX - Topic"),
    "사계": ("One Day Only (사계) (하루살이)", "M.C the MAX - Topic"),
    "어디에도": ("No matter where (어디에도)", "M.C the MAX - Topic"),
    "시간을 견디면": ("Time Will Tell (시간을 견디면)", "M.C the MAX - Topic"),
    "넘쳐흘러": ("After You've Gone (넘쳐흘러)", "M.C the MAX - Topic"),

    # ==========================================
    # 전기뱀장어 (The Electriceels) - 한글 제목
    # 데이터에 이미 한글로 저장되어 있음
    # ==========================================

    # ==========================================
    # 델리스파이스 (Deli Spice)
    # ==========================================
    "챠우챠우": ("chow chow-No matter how hard I try to block it, I can hear your voice (챠우챠우-아무리...", "Deli Spice - Topic"),

}

# ==========================================
# 역방향: 데이터 속 영어 제목 → 사용자 친화적 한국어 제목
# ==========================================
ENGLISH_TO_KOREAN_DISPLAY = {
    # 검정치마
    "Holiday": "빨간 나를 (Holiday)",
    "Wrong question": "틀린질문 (Wrong Question)",
    "Island (queen of diamonds)": "섬 (Island)",
    "Sangsu station": "상수역 (Sangsu Station)",
    "Mad dog diary": "광견일기 (Mad Dog Diary)",
    "Hawaiian black sand": "하와이 검은 모래 (Hawaiian Black Sand)",
    "Thinner than water": "맑고 묽게 (Thinner Than Water)",
    "My shadow": "그늘은 그림자로 (My Shadow)",
    "Blood and thirst (king of hurts)": "피와 갈증 (Blood and thirst)",
    "Baptized In Fire": "불세례 (Baptized In Fire)",
    "My Little Lambs": "어린양 (My Little Lambs)",
    "Cicadas": "매미들 (Cicadas)",
    "Follow You": "따라갈래 (Follow You)",
    "Min": "미는 남자 (Min)",
    "Electra": "일렉트라 (Electra)",
    "Powder Blue": "파우더 블루 (Powder Blue)",
    "Antifreeze": "안티프리즈 (Antifreeze)",
    "Love Is All": "사랑은 모든 것 (Love Is All)",
    "EVERYTHING": "모든 것 (EVERYTHING)",
    "Puppy": "강아지 (Puppy)",
    "Big Love": "빅 러브 (Big Love)",
    "Friends In Bed": "프렌즈 인 베드 (Friends In Bed)",
    "Ling Ling": "링 링 (Ling Ling)",
    "Love Shine": "러브 샤인 (Love Shine)",
    "Sunday Girl": "선데이 걸 (Sunday Girl)",
    "Bollywood": "발리우드 (Bollywood)",
    "Flying Bobs": "플라잉 밥스 (Flying Bobs)",

    # 잔나비
    "See Your Eyes": "씨 유어 아이즈 (See Your Eyes)",
    "Surprise!": "서프라이즈! (Surprise!)",
    "Good Boy Twist": "굿 보이 트위스트 (Good Boy Twist)",
    "JUNGLE": "정글 (JUNGLE)",
    "pony (pony)": "포니 (pony)",
    "November Rain": "노벰버 레인 (November Rain)",
    "Wish": "위시 (Wish)",
    "LEGEND (전설)": "전설 (LEGEND)",
    "Summer ll (밤의 공원)": "밤의 공원 (Summer ll)",
    "Sweet memories (그 밤 그 밤)": "그 밤 그 밤 (Sweet memories)",
    "A Ballad of Non Le Jon (비틀 파워!)": "비틀 파워! (A Ballad of Non Le Jon)",
    
    # 카더가든 (Car, the Garden)
    "Dreamed a dream": "꿈을 꿨어요 (Dreamed a dream)",
    "Tomorrow": "내일 (Tomorrow)",
    "No place to hide": "숨을 곳이 없어요 (No place to hide)",
    "Just as We Were": "우리가 있던 그대로 (Just as We Were)",
    "My whole world": "나의 온 세상 (My whole world)",
    "Ta-Ta For Now (이젠 안녕)": "이젠 안녕 (Ta-Ta For Now)",
    
    # Xdinary Heroes
    "Money On My Mind": "머니 온 마이 마인드 (Money On My Mind)",
    "Strawberry Cake": "스트로베리 케이크 (Strawberry Cake)",
    "Happy Death Day": "해피 데스 데이 (Happy Death Day)",
    
    # M.C the MAX
    "One Day Only (사계) (하루살이)": "하루살이 (One Day Only)",
    "Lying On Your Lips (입술의 말)": "입술의 말 (Lying On Your Lips)",
    "No matter where (어디에도)": "어디에도 (No matter where)",
    "Time Will Tell (시간을 견디면)": "시간을 견디면 (Time Will Tell)",
    "After You've Gone (넘쳐흘러)": "넘쳐흘러 (After You've Gone)",
    "BLOOM (처음처럼)": "처음처럼 (BLOOM)",
    
    # CNBLUE
    "Intuition (직감)": "직감 (Intuition)",
    "Can't Stop (Can't Stop)": "Can't Stop",
    
    # Oasis, Charlie Puth (영어가 익숙하므로 기본 유지하되 특정 곡들만 보기 좋게)
    "Morning Glory (Live at Knebworth)": "Morning Glory (Live)",
    "Roll with It (Live at Knebworth)": "Roll with It (Live)",
    "We Don't Talk Anymore": "We Don't Talk Anymore",
}


def search_song(keyword, artist_hint=None, df=None):
    """
    사용자가 한글로 입력해도 데이터에서 올바르게 찾아주는 통합 검색 함수
    """
    if df is None:
        import pandas as pd
        df = pd.read_csv(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv')

    # 1. alias에서 먼저 찾기
    if keyword in TITLE_ALIAS:
        eng_title, artist = TITLE_ALIAS[keyword]
        results = df[
            df['title'].str.contains(eng_title, case=False, na=False) &
            df['artist'].str.contains(artist.replace(' - Topic', ''), case=False, na=False)
        ]
        if len(results) > 0:
            song_id = results['song_id'].value_counts().index[0]
            print(f"[Alias 매핑] '{keyword}' → '{song_id}' ({len(results)}회)")
            return song_id

    # 2. 직접 제목 검색 (한글/영어 무관)
    results = df[df['title'].str.contains(keyword, case=False, na=False)]
    if artist_hint:
        results = results[results['artist'].str.contains(artist_hint, case=False, na=False)]

    if len(results) > 0:
        song_id = results['song_id'].value_counts().index[0]
        print(f"[직접 검색] '{keyword}' → '{song_id}' ({len(results)}회)")
        return song_id

    print(f"[검색 실패] '{keyword}' 를 데이터에서 찾을 수 없습니다.")
    return None


if __name__ == "__main__":
    import pandas as pd
    import codecs, sys
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

    df = pd.read_csv(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv')

    print("=== 제목 alias 검색 테스트 ===\n")
    test_keywords = [
        "빨간 나를",
        "해초",
        "광견일기",
        "어린양",
        "하루살이",
        "After School Activity",
        "See Your Eyes",
        "1:05",
    ]
    for kw in test_keywords:
        search_song(kw, df=df)
