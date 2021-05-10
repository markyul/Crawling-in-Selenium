import cv2
import glob
from urllib.request import urlretrieve  # 이미지 다운로드에 필요
from mtcnn import MTCNN

# 사진 검출기
def imgDetector(img):
    global index
    global blank_num
    global minus_num
    filetype = ".jpg"
    # 영상 압축
    # img = cv2.resize(img, dsize=None, fx=1.5, fy=1.5)

    # 그레이 스케일 변환
    change_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detector = MTCNN()
    # MTCNN 얼굴 탐지 알고리즘
    results = detector.detect_faces(change_img)

    print(len(results))
    if len(results) == 0:
        blank_num = blank_num + 1
    for faces in results:
        x, y, w, h = faces["box"]
        try:
            crop_img = img[y : y + h, x : x + w].copy()
            # cv2.imwrite("../test_images/{}{}".format(index, filetype), crop_img)
            cv2.imwrite(
                "C:/Users/HakRyul/Desktop/PBL_asian_face_copy_crop/{}{}".format(
                    index, filetype
                ),
                crop_img,
            )
            index = index + 1
        except:
            minus_num = minus_num + 1


index = 0
minus_num = 0
imgs = []

print("started..")
# 폴더에서 이미지 가져와서 저장하기
for file in glob.glob("C:/Users/HakRyul/Desktop/PBL_asian_face_copy/*.jpg"):
    imgs.append(cv2.imread(file))

print("imgs loaded..")

errList = []
blank_num = 0
for img in imgs:
    try:
        imgDetector(img)  # 사진 탐지기

    except Exception as e:
        # print(e)
        errList.append(e)


print("completed..")

# test_img = cv2.imread("2.jpg")
# imgDetector(test_img)  # 사진 탐지기

print(len(imgs))  # 총 사진 수
print(index)  # 얼굴 수
print(blank_num)  # 얼굴이 탐지 되지 않은 사진 수
print(minus_num)  # 좌표가 잘못 찍혀서 예외 처리된 얼굴 수
print(errList)