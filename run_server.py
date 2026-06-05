"""서버 시작 + 준비되면 브라우저 자동 오픈"""
import subprocess, sys, threading, time, webbrowser, urllib.request, os

def wait_and_open():
    """서버가 응답할 때까지 대기 후 브라우저 열기"""
    for _ in range(60):
        time.sleep(2)
        try:
            urllib.request.urlopen('http://localhost:5000', timeout=2)
            print("\n  >>> 서버 준비 완료! 브라우저를 엽니다...\n")
            webbrowser.open('http://localhost:5000')
            return
        except:
            pass
    print("\n  >>> 서버 시작 시간 초과")

# 브라우저 오픈 스레드
threading.Thread(target=wait_and_open, daemon=True).start()

# 서버 실행
os.chdir(os.path.dirname(os.path.abspath(__file__)))
subprocess.run([sys.executable, 'app.py'])
