import cv2 as cv
import numpy as np
import math
import csv
capture = cv.VideoCapture('C:/Users/Admin/Desktop/FallDetection/Fall/Fall (1).avi')
backSub = cv.createBackgroundSubtractorKNN()
x1 = 0
y1 = 0
BG = None
count = 0
rowlist = []
Do_lech_chuan = 0.0
Toc_do_thay_doi_trong_tam = 0.0
mang = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
mang1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
k = 0
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    frame1 = backSub.apply(frame)
    frame1 = cv.GaussianBlur(frame1, (5, 5), 2, sigmaY=2)
    thresh = cv.threshold(frame1, 130, 255, cv.THRESH_BINARY)[1]
    kernels = np.ones((4,4), np.uint8)
    kernel = np.array([[0, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 0]], dtype=np.uint8)
    thresh = cv.erode(thresh, kernels, iterations=1)
    thresh = cv.dilate(thresh, kernel, iterations=3)
    thresh = cv.erode(thresh, kernels, iterations=1)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if count < 4:
      print("bo hinh")
    else:
        for contour in contours:
          (x, y), (major, minor), angle = cv.fitEllipse(contour)
          print("center x,y:",x,y)
          print("canh dai rong:",major,minor)
          print("goc:",angle)
          if cv.contourArea(contour) < 900:
            continue
          cv.ellipse(frame, ((x,y), (major,minor), angle), (0,255,0), 2)

          x2 = x1 - x
          y2 = y1 - y
          a = pow(y, 2)/pow(x, 2)
          e = math.sqrt(abs(1 - a))
          v0 = math.sqrt(pow(x2, 2) + pow(y2, 2))
          x1 = x
          y1 = y
          print("vantoc", v0)
          print ("e la", e)
          if(k<10):
            mang[k] = angle
            k +=1
          if(k==10):
            for i in range(9):
              Do_lech_chuan = sum(mang)/10
              mang[i] = mang[i+1]
              mang[9] = angle
              print("do lech chuan la:", Do_lech_chuan)

          if(k<10):
            mang1[k] = y
            k +=1
          if(k==10):
            for i in range(9):
              Toc_do_thay_doi_trong_tam = sum(mang1)/10
              mang1[i] = mang1[i+1]
              mang1[9] = y
              print("toc do thay doi trong tam la:", Toc_do_thay_doi_trong_tam)
          rowlist = ["%f" % v0, "%f" % angle, "%f" % e ,"%f" % Do_lech_chuan, "%f" % Toc_do_thay_doi_trong_tam]
          if rowlist:
            with open('C:/Users/Admin/Desktop/FallDetection/Non_CSV/8_2.csv', 'a', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(rowlist)


        frame = cv.putText(frame, 'frame: %d' %count, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
        cv.imshow("ve", frame) 
        cv.imshow('frame',thresh)
            
    count += 1
    if cv.waitKey(30) == 27:
        break

capture.release()
cv.destroyAllWindows()