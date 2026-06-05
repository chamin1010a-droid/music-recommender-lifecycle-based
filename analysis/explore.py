import os
from bs4 import BeautifulSoup

file_path = r"c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\시청 기록.html"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        # File is large, we can't load the whole file into memory if it's 41MB, but 41MB is actually fine for BeautifulSoup.
        soup = BeautifulSoup(f.read(), 'html.parser')

    # Look for 'content-cell' which is a typical Google Takeout container
    cells = soup.find_all('div', class_='content-cell')
    print(f"Total content-cell count: {len(cells)}")
    
    # Let's print the first 5 cells that mention "YouTube Music"
    count = 0
    for cell in cells:
        text = cell.get_text(separator=' | ', strip=True)
        if 'YouTube Music' in text:
            print(f"--- Entry {count+1} ---")
            print(text)
            count += 1
            if count >= 5:
                break
                
except Exception as e:
    print(f"Error: {e}")
