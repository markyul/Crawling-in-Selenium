from selenium import webdriver
import time

# selenium에서 사용할 웹 드라이버 절대 경로 정보
chromedriver = "C:\chromedriver\chromedriver.exe"
# selenum의 webdriver에 앞서 설치한 chromedirver를 연동한다.
driver = webdriver.Chrome(chromedriver)
# driver로 특정 페이지를 크롤링한다.
driver.get("https://auto.naver.com/bike/mainList.nhn")

print("+" * 100)
print(driver.title)  # 크롤링한 페이지의 title 정보
print(driver.current_url)  # 현재 크롤링된conda  페이지의 url
print("바이크 브랜드 크롤링")
print("-" * 100)
# 위의 구조가 기본 시작

# 바이크 제조사 전체 페이지 버튼 클릭
bikeCompanyAllBtn = driver.find_element_by_css_selector(
    "#container > div.spot_main > div.spot_aside > div.tit > a"
)
bikeCompanyAllBtn.click()

# time.sleep(3)

# 바이크 제조사 1페이지 리스트
bikeCompanyList = driver.find_elements_by_css_selector(
    "#_vendor_select_layer > div > div.maker_group > div.emblem_area > ul > li"
)

# 1페이지 크롤링
for item in bikeCompanyList:
    bikeCompanyName = item.find_element_by_tag_name("span").text
    if bikeCompanyName != "":
        print("바이크 제조사명: " + bikeCompanyName)
        ahref = item.find_element_by_tag_name("a").get_attribute("href")
        print("네이버 자동차 바이크제조사 홈 sub url:", ahref)
        imgUrl = item.find_element_by_tag_name("img").get_attribute("src")
        print("바이크 회사 엠블럼: " + imgUrl)

time.sleep(3)

nextBtn = driver.find_element_by_css_selector(
    "#_vendor_select_layer > div > div.maker_group > div.rolling_btn > button.next"
)
# 다음 페이지가 존제하는지 다음 버튼 활성화 여부로 확인
isExistNextPage = nextBtn.is_enabled()

if isExistNextPage:
    print("다음 페이지 ==============================")
    nextBtn.click()
    bikeCompanyList = driver.find_elements_by_css_selector(
        "#_vendor_select_layer > div > div.maker_group > div.emblem_area > ul > li"
    )

    for item in bikeCompanyList:
        bikeCompanyName = item.find_element_by_tag_name("span").text
        if bikeCompanyName != "":
            print("바이크 제조사명: " + bikeCompanyName)
            ahref = item.find_element_by_tag_name("a").get_attribute("href")
            print("네이버 자동차 바이크제조사 홈 sub url:", ahref)
            imgUrl = item.find_element_by_tag_name("img").get_attribute("src")
            print("바이크 회사 엠블럼: " + imgUrl)
