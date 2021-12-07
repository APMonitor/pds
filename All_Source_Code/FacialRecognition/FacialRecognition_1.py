import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
import urllib.request

# download image as class.jpg
url = 'http://apmonitor.com/pds/uploads/Main/students_walking.jpg'
urllib.request.urlretrieve(url, 'class.jpg')

def draw_faces(data, result_list):
    for i in range(len(result_list)):
        x1, y1, width, height = result_list[i]['box']
        x2, y2 = x1 + width, y1 + height
        plt.subplot(1, len(result_list), i+1)
        plt.axis('off')
        plt.imshow(data[y1:y2, x1:x2])
    plt.show()

pixels = plt.imread('class.jpg')      # read image
detector = MTCNN()                    # create detector
faces = detector.detect_faces(pixels) # detect faces
draw_faces(pixels, faces)             # display faces