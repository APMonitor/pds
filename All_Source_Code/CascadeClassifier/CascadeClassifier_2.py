import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import urllib.request

# Get camera and display video
camera = cv2.VideoCapture(0)
WindowName = 'Face Detection'
cv2.namedWindow(WindowName, cv2.WINDOW_AUTOSIZE)

# download cascade classifier configuration
url = 'http://apmonitor.com/pds/uploads/Main/cascade.xml'
urllib.request.urlretrieve(url, 'cascade.xml')

faceCascade = cv2.CascadeClassifier('cascade.xml')

start = time.time()
while time.time()-start<=20.0:
    # read frame
    ret, pixels = camera.read()         
    # convert to gray scale and identify faces
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,
                                         minNeighbors=3,\
                                         minSize=(30, 30))
    # display identified faces on original image
    for (x, y, w, h) in faces:
        cv2.rectangle(pixels,(x,y),(x+w,y+h),(255,255,0),3)
    cv2.imshow(WindowName,pixels)        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and video file
camera.release()
cv2.destroyAllWindows()