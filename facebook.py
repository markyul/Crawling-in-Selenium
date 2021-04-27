from selenium import webdriver
import os
import time
from datetime import datetime, timedelta
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from urllib.request import urlretrieve  # 이미지 다운로드에 필요

minute_count = 0  # 시간을 세기 위한 선언

# 시간 세기
def time_count(start_time):
    global minute_count
    current_time = datetime.now()

    time_diff = current_time - start_time
    minute = (int)(time_diff.seconds / 60)
    if minute_count == minute:
        print(minute, "분")
        minute_count = minute_count + 1


# 스크롤 내리기
def scroll():
    start_time = datetime.now()
    global minute_count
    minute_count = 0
    while minute_count != 2:
        # 끝까지 스크롤 다운
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time_count(start_time)


img_folder_path = "D:/University/전공/4학년/문제해결실무/images"
options = Options()
tags = ["증명사진", "セルカ", "tựsướng"]

# selenium에서 사용할 웹 드라이버 절대 경로 정보
chromedriver = "C:\chromedriver\chromedriver.exe"
# 팝업 권한 허용
prefs = {
    "profile.default_content_setting_values.notifications": 1
}  # 1이 허용, #2가 금지인 거 같음
options.add_experimental_option("prefs", prefs)
# selenum의 webdriver에 앞서 설치한 chromedirver를 연동한다.
driver = webdriver.Chrome(chromedriver, chrome_options=options)
# driver로 특정 페이지를 크롤링한다.
driver.get("https://www.facebook.com/")
# chrome을 전체화면으로 넓히는 옵션입니다.
options.add_argument("--start-fullscreen")

print("+" * 100)
print(driver.title)  # 크롤링한 페이지의 title 정보
print(driver.current_url)  # 현재 크롤링된conda  페이지의 url
print("페이스북 크롤링")
print("-" * 100)

driver.find_element_by_name("email").send_keys("pig052656@naver.com")
driver.find_element_by_name("pass").send_keys("Rjal6811!!")
driver.find_element_by_name("login").submit()

for tag in tags:
    result = []  # 마지막 이미지 주소 저장
    time.sleep(3)

    # 홈으로 이동
    driver.find_element_by_xpath(
        "//div[1]/div/div[1]/div/div[2]/div[1]/a"
    ).click()
    time.sleep(3)

    driver.find_element_by_xpath(
        "//div[1]/div/div[1]/div/div[2]/div[2]/div/div/div[1]/div/div/label/input"
    ).send_keys(tag)
    time.sleep(3)

    driver.find_element_by_xpath(
        "//div[1]/div/div[1]/div/div[2]/div[2]/div/div/div[1]/div/div/label/input"
    ).send_keys(Keys.RETURN)
    time.sleep(3)

    # 사진 카테고리 클릭
    driver.find_element_by_xpath(
        "//div[1]/div/div[1]/div/div[3]/div/div/div/div[1]/div[1]/div/div[2]/div[1]/div[2]/div/div/div[2]/div[4]/a"
    ).click()
    time.sleep(3)

    # //*[@id="mount_0_0_tn"]/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div[1]/div/div[2]/div[1]/div[2]/div/div/div[2]/div[4]/a
    # //*[@id="mount_0_0_tn"]/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div[1]/div/div[2]/div[1]/div[2]/div/div/div[2]/div[4]/a
    # driver.find_element_by_xpath(
    #     "//div[1]/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div[2]/div/div/div/div/div/div/div[2]/div/div/div/div/div[2]/a"
    # ).click()

    # 모든 사진 보기
    driver.find_element_by_css_selector(
        "div:nth-child(1) > div > div:nth-child(1) > div > div.rq0escxv.l9j0dhe7.du4w35lb > div > div > div.j83agx80.cbu4d94t.d6urw2fd.dp1hu0rb.l9j0dhe7.du4w35lb > div.rq0escxv.l9j0dhe7.du4w35lb.j83agx80.pfnyh3mw.jifvfom9.gs1a9yip.owycx6da.btwxx1t3.buofh1pr.dp1hu0rb.ka73uehy > div.rq0escxv.l9j0dhe7.du4w35lb.j83agx80.cbu4d94t.g5gj957u.d2edcug0.hpfvmrgz.rj1gh0hx.buofh1pr.dp1hu0rb > div > div > div > div > div > div > div:nth-child(2) > div > div > div > div > div.dhix69tm.sjgh65i0.wkznzc2l.tr9rh885 > a"
    ).click()
    time.sleep(5)
    scroll()

    imgs = driver.find_elements_by_css_selector(
        "div:nth-child(1) > div > div:nth-child(1) > div > div.rq0escxv.l9j0dhe7.du4w35lb > div > div > div.j83agx80.cbu4d94t.d6urw2fd.dp1hu0rb.l9j0dhe7.du4w35lb > div.rq0escxv.l9j0dhe7.du4w35lb.j83agx80.pfnyh3mw.jifvfom9.gs1a9yip.owycx6da.btwxx1t3.buofh1pr.dp1hu0rb.ka73uehy > div.rq0escxv.l9j0dhe7.du4w35lb.j83agx80.cbu4d94t.g5gj957u.d2edcug0.hpfvmrgz.rj1gh0hx.buofh1pr.dp1hu0rb > div > div > div > div > div > div:nth-child(1) > div > div > div > a > div > img"
    )

    for img in imgs:  # 모든 이미지들을 탐색
        result.append(img.get_attribute("src"))  # 이미지 src만 모아서 리스트에 저장

    print(len(result))
    for index, link in enumerate(
        result
    ):  # 리스트에 있는 원소만큼 반복, 인덱스는 index에, 원소들은 link를 통해 접근 가능
        filetype = ".jpg"  # 확장자명을 잘라서 filetype변수에 저장 (ex -> .jpg)
        urlretrieve(
            link,
            "D:/University/전공/4학년/문제해결실무/images/{}{}{}".format(
                tag, index, filetype
            ),
        )  # link에서 이미지 다운로드, './imgs/'에 파일명은 index와 확장자명으로
