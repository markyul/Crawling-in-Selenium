import cv2
import glob
from urllib.request import urlretrieve  # 이미지 다운로드에 필요

minus_num = 0

# 사진 검출기
def imgFilter(img):
    global index

    filetype = ".jpg"
    cv2.imwrite(
        "C:/Users/HakRyul/Desktop/result_crop_img/{}{}".format(index, filetype),
        img,
    )
    index = index + 1


imgs = []
# 폴더에서 이미지 가져와서 저장하기
for file in glob.glob(
    "C:/Users/HakRyul/Desktop/PBL_asian_face_copy_crop/*.jpg"
):
    imgs.append(cv2.imread(file))

stdW = 100
stdH = 100
index = 0
deletedCnt = 0

for img in imgs:
    h, w, c = img.shape

    """
    print('width:  ', w)
    print('height: ', h)
    print('channel:', c)
    """

    if h < stdW or w < stdH:
        deletedCnt += 1
        continue

    imgFilter(img)
