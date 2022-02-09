import cv2 as cv
import matplotlib.pyplot as plt
im = cv.imread('students_walking.jpg')
im2 = plt.imread('students_walking.jpg')
plt.subplot(1,2,1)
plt.imshow(im)
plt.title('BGR (OpenCV)')
plt.subplot(1,2,2)
plt.imshow(im2)
plt.title('RGB (Matplotlib)')