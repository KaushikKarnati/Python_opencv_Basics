import cv2
import numpy as np

'''
#chapter 1 - Intro to opencv
//code for reading images
img = cv2.imread("Resources/pikachu.jpg")
cv2.imshow("output", img)
cv2.waitKey()
'''
'''
//for playing videos
cap = cv2.VideoCapture("Resources/videoplayback.mp4")
while True:
    success, img = cap.read()
    cv2.imshow("video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''
'''
cap = cv2.VideoCapture(0)
#id's and values
cap.set(3,640)
cap.set(4,480)
cap.set(10,100)
while True:
    success , img = cap.read()
    cv2.imshow("web_cam",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''
'''
//basic functions in opencv
#chapter 2 - image manipulation
k = np.ones((5,5),np.uint8())
img = cv2.imread("Resources/pikachu.jpg")
img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_blur = cv2.blur(img_grey,(7,7),0)
img_canny = cv2.Canny(img_grey,100,100)
img_dialation = cv2.dilate(img_canny, k , iterations=5)
img_erd = cv2.erode(img_dialation,k,iterations=4)



while True:
    cv2.imshow("Grey_Image", img_erd)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

'''

'''
#cropping and resizing images - chapter 3
img = cv2.imread('Resources/squirtle.jpg')
img_resize = cv2.resize(img,(500,500),3)
#img[height,width]
img_crop = img[0:200,200:250]
#print(img_resize.shape)
while True:
    cv2.imshow("resize",img_crop)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''

'''
# chapter 4 - shapes and texts

img = np.zeros((512, 512, 3), np.uint8)
cv2.line(img, (0, 0), (512, 512), (0, 255, 0), 3)  # line (source,(starting point),(ending point),(colour),thickness)
cv2.rectangle(img, (0, 0), (200, 300), (0, 0, 255), 2)  # rectangle FILLED-fill colour
cv2.circle(img, (256, 256), 50, (255, 0, 0), 2)  # circle
cv2.putText(img, "kaushik", (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),1)  # text (source, origin_point, font,scale,thickness
# img[:] = 255,0,0 #for colouring full image

while True:
    cv2.imshow("Draw_Img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''

'''
# chapter 5 - warp perspective
img = cv2.imread('Resources/image.JPG')
width, height = 250, 350
pt1 = np.float32([[111, 219], [287, 188], [154, 482], [352, 450]])
pt2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
matrix = cv2.getPerspectiveTransform(pt1, pt2)
img_output = cv2.warpPerspective(img, matrix, (width, height))
while True:
    cv2.imshow('warp', img)
    cv2.imshow('warped', img_output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''

'''
# chapter 6 joining images

img = cv2.imread('Resources/bulbasaur.jpg')
hor_img = np.hstack((img,img)) # stack images for joining them together in horizontal
ver_img = np.vstack((img,img)) # in vertical position

while True:
    cv2.imshow('image',hor_img)
    cv2.imshow('image_ver',ver_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''

'''
# chapter 7 detecting colours
def empty(a):
    pass


cv2.namedWindow('TrackBar') # creating a window 
cv2.resizeWindow('TrackBar', 640, 480) # resizing the window 
cv2.createTrackbar('Hue_Min', 'TrackBar', 0, 179, empty) # creating sliding bars for hue
cv2.createTrackbar('Hue_Max', 'TrackBar', 179, 179, empty) # creating sliding bars hue_max
cv2.createTrackbar('Sat_Min', 'TrackBar', 116, 255, empty) # creating sliding bars saturation_min
cv2.createTrackbar('Sat_Max', 'TrackBar', 255, 255, empty) # creating sliding bars saturation_max
cv2.createTrackbar('val_Min', 'TrackBar', 1, 255, empty) # creating sliding bars Value_min
cv2.createTrackbar('val_Max', 'TrackBar', 255, 255, empty) # creating sliding bars Value_max

while True:
    img = cv2.imread('Resources/image.JPG')

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos('Hue_min', 'TrackBar') # using the created slide bars
    h_max = cv2.getTrackbarPos('Hue_Max', 'TrackBar') # using the created slide bars
    s_min = cv2.getTrackbarPos('Sat_Min', 'TrackBar') # using the created slide bars
    s_max = cv2.getTrackbarPos('Sat_Max', 'TrackBar') # using the created slide bars
    v_min = cv2.getTrackbarPos('val_Min', 'TrackBar') # using the created slide bars
    v_max = cv2.getTrackbarPos('val_Max', 'TrackBar') # using the created slide bars
    lower = np.array([h_min, s_min, v_min]) # using numpy to create the value arrays
    upper = np.array([h_max, s_max, v_max]) # upper and lower

    masked = cv2.inRange(hsv_img, lower, upper) 
    img_result = cv2.bitwise_and(img,img,mask=masked) # joining the masked image to the original 
    print(h_min,h_max,s_min,s_max,v_min,v_max) # -1 105 156 255 151 255 value for the required colour
    cv2.imshow('image', img)
    # cv2.imshow('Hsv', hsv_img)
    cv2.imshow('masked', masked)
    cv2.imshow('result_image',img_result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''


# chapter 8 detecting shapes

def Stacked_Images(scale, imgArray):  # function to stack images
    rows = len(imgArray)
    cols = len(imgArray[0])
    Available_Rows = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if Available_Rows:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        image_Blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_Blank] * rows
        hor_con = [image_Blank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area > 500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            # print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if objCor == 3:
                objectType = "Tri"
            elif objCor == 4:
                aspRatio = w / float(h)
                if 0.98 < aspRatio < 1.03:
                    objectType = "Square"
                else:
                    objectType = "Rectangle"
            elif objCor > 4:
                objectType = "Circles"
            else:
                objectType = "None"

            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imgContour, objectType,
                        (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 0, 0), 2)


img = cv2.imread('Resources/circle.png')
imgContour = img.copy()

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
imgCanny = cv2.Canny(imgBlur, 50, 50)
getContours(imgCanny)

imgBlank = np.zeros_like(img)
imgStack = Stacked_Images(0.8, ([img, imgGray, imgBlur],
                                [imgCanny, imgContour, imgBlank]))

cv2.imshow("Stack", imgStack)
cv2.waitKey(0)

