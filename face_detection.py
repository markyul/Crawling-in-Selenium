import cv2
import glob
import time
import os
from urllib.request import urlretrieve  # 이미지 다운로드에 필요

# 사진 검출기
def imgDetector(img, cascade):
    test_img = cv2.imread("test.jpg")
    img_type = type(test_img)
    global num
    global index
    global blank_num
    filetype = ".jpg"
    # 영상 압축
    # img = cv2.resize(img, dsize=None, fx=1.5, fy=1.5)

    # if type(img) != img_type:
    #     num = num + 1
    #     cv2.imshow("hi", img)
    #     time.sleep(100)
    #     return

    # 그레이 스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cascade 얼굴 탐지 알고리즘
    results = cascade.detectMultiScale(
        img,  # 입력 이미지
        scaleFactor=1.5,  # 이미지 피라미드 스케일 factor
        minNeighbors=5,  # 인접 객체 최소 거리 픽셀
        # 탐지 객체 최소 크기
    )
    if len(results) == 0:
        blank_num = blank_num + 1
    for box in results:
        x, y, w, h = box
        crop_img = img[y : y + h, x : x + w].copy()
        # cv2.imshow("hi", crop_img)
        cv2.imwrite("../crop_images/{}{}".format(index, filetype), crop_img)
        # crop_img.save("{}{}".format(index, filetype), "JPEG")
        index = index + 1

        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

    # for save in result:
    #     print(save.shape)
    #     index = index + 1
    #     cv2.imwrite(
    #         "D:/University/전공/4학년/문제해결실무/crop_images/{}{}".format(
    #             index, filetype
    #         ),
    #         save,
    #     )

    # 사진 출력
    # cv2.imshow("facenet", img)
    # cv2.waitKey(10000)


index = 0
# 가중치 파일 경로
cascade_filename = "haarcascade_frontalface_default.xml"
# 모델 불러오기
cascade = cv2.CascadeClassifier(cascade_filename)

# 이미지 파일
# imgs = [
#     cv2.imread(file)
#     for file in glob.glob("C:/Users/HakRyul/Desktop/images/*.jpg")
# ]
imgs = []

test_img = cv2.imread("test.jpg")
img_type = type(test_img)

# 폴더에서 이미지 가져와서 저장하기
for file in sorted(glob.glob("C:/Users/HakRyul/Desktop/images_copy/*.jpg")):
    imgs.append(cv2.imread(file))

print(len(imgs))


num = 0
blank_num = 0
for img in imgs:
    imgDetector(img, cascade)  # 사진 탐지기

print(index)
print(num)
print(blank_num)
# test_img = cv2.imread("test.jpg")
# imgDetector(test_img, cascade, index)  # 사진 탐지기
