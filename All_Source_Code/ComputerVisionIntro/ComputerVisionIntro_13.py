import cv2 as cv
import time

# Get camera Object
camera = cv.VideoCapture(0)                         
w = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))    
h = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))

# Write video.avi
out = cv.VideoWriter('video.avi', \
                     cv.VideoWriter_fourcc(*'XVID'), \
                     25, (w,h))

# Create Window to video frames
WindowName = 'View'
cv.namedWindow(WindowName, cv.WINDOW_AUTOSIZE)

# Save and view 5 second video
start = time.time()
while time.time()-start<=5.0:
    ret0, frame = camera.read()         
    cv.imshow(WindowName, frame)        
    out.write(frame)                    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.imwrite('frame.jpg', frame)          

# Release camera and video file
camera.release(); out.release()
cv.destroyAllWindows()