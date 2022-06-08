import cv2
import numpy as np
import math
import csv

video = cv2.VideoCapture('F:/OpenCV Project/secondHello/Fall (6).avi')
x1 = 0
y1 = 0
BG = None
count = 0
rowlist = []
while video.isOpened():
    ret, frame = video.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (5, 5), 2, sigmaY=2)
        if BG is None:
            BG = frame
        else:
            
            frameDelta = cv2.absdiff(BG, frame)
            frameDelta += abs(np.min(frameDelta))
            thresh = cv2.threshold(frameDelta, 45, 255, cv2.THRESH_BINARY)[1]
            
            kernels = np.ones((4,4), np.uint8)
            kernel = np.array([[0, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 0]], dtype=np.uint8)
            
            thresh = cv2.erode(thresh, kernels, iterations=3)
            thresh = cv2.dilate(thresh, kernel, iterations=10)
            thresh = cv2.erode(thresh, kernels, iterations=3)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
              (x, y), (major, minor), angle = cv2.fitEllipse(contour)
              print("center x,y:",x,y)
              print("canh dai rong:",major,minor)
              print("goc:",angle)
              if cv2.contourArea(contour) < 800:
                continue
              cv2.ellipse(frame, ((x,y), (major,minor), angle), (0,255,0), 2)

              x2 = x1 - x
              y2 = y1 - y
              a = pow(y, 2)/pow(x, 2)
              e = math.sqrt(abs(1 - a))
              v0 = math.sqrt(pow(x2, 2) + pow(y2, 2))
              x1 = x
              y1 = y
              print("vantoc", v0)
              print ("e la", e)
              rowlist = ["%f" % v0, "%f" % angle, "%f" % x, "%f" % y, "%f" % e ,"%f" % count ]
              if rowlist:
                 with open('ex30.csv', 'a', newline='') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(rowlist)
            count += 1
            frame = cv2.putText(frame, 'frame: %d' %count, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("ve", frame) 
            cv2.imshow('frame',thresh)
        if cv2.waitKey(30) == 27:
            break
    else:
        break

video.release()
cv2.destroyAllWindows()