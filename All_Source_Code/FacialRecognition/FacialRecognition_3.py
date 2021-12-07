import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import urllib.request

# download image as class.jpg
url = 'http://apmonitor.com/pds/uploads/Main/students_walking.jpg'
urllib.request.urlretrieve(url, 'class.jpg')

# download cascade classifier configuration
url = 'http://apmonitor.com/pds/uploads/Main/cascade.xml'
urllib.request.urlretrieve(url, 'cascade.xml')

def draw_faces(data, result_list):
    for i in range(len(result_list)):
        x1, y1, width, height = result_list[i]
        x2, y2 = x1 + width, y1 + height
        plt.subplot(1, len(result_list), i+1)
        plt.axis('off')
        plt.imshow(data[y1:y2, x1:x2])

pixels = plt.imread('class.jpg')
faceCascade = cv2.CascadeClassifier('cascade.xml')
gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,
                                     minNeighbors=2,\
                                     minSize=(10, 10))

# display only the faces
draw_faces(pixels, faces)             

# display identified faces on original image
fig, ax = plt.subplots(); ax.imshow(pixels)
for (x, y, w, h) in faces:
    rect = patches.Rectangle((x, y), w, h, lw=2, \
                             alpha=0.5, edgecolor='r', \
                             facecolor='none')
    ax.add_patch(rect)

plt.show()