from selenium import webdriver
import time

# selenium에서 사용할 웹 드라이버 절대 경로 정보
chromedriver = "C:\chromedriver\chromedriver.exe"
# selenum의 webdriver에 앞서 설치한 chromedirver를 연동한다.
driver = webdriver.Chrome(chromedriver)
# driver로 특정 페이지를 크롤링한다.
driver.get(
    "https://www.instagram.com/explore/tags/%EC%A6%9D%EB%AA%85%EC%82%AC%EC%A7%84/"
)

print("+" * 100)
print(driver.title)  # 크롤링한 페이지의 title 정보
print(driver.current_url)  # 현재 크롤링된conda  페이지의 url
print("인스타 크롤링")
print("-" * 100)
time.sleep(5)
driver.find_element_by_name("username").send_keys("01047535787")
driver.find_element_by_name("password").send_keys("rjal6811!!")
time.sleep(2)
driver.find_element_by_xpath('//*[@id="loginForm"]/div/div[3]/button').submit()

time.sleep(3)
driver.get(
    "https://www.instagram.com/explore/tags/%EC%A6%9D%EB%AA%85%EC%82%AC%EC%A7%84/"
)
time.sleep(3)

# 사진 리스트
instaImgList = driver.find_elements_by_css_selector(
    "#react-root > section > main > article > div:nth-child(3) > div > div > div"
)
time.sleep(3)

for item in instaImgList:
    print(
        item.get_attribute("class")
    )  # 사진 하나하나가 출력되기는 하는데 24개만 추출됨 스크롤을 내려야하나?
    # print(len(imgList))
    # for data in imgList:
    #     print(data.get_attribute("class"))