import threading
import numpy as np
import cv2 as cv
from PIL import Image, ImageTk
from tkinter import Tk, Frame, Button, BOTH, Label, Scale, Radiobutton       # Graphical User Inetrface Stuff
from tkinter import font as tkFont
import tkinter as tk

camera = cv.VideoCapture(0)
width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
videoout = cv.VideoWriter('./Video.avi', cv.VideoWriter_fourcc(*'XVID'), 25, (width, height))   # Video format

# Button Definitions
ORIGINAL = 0
BINARY = 1
EDGE = 2
LINE = 3
ABSDIFF = 4
RGB = 5
HSV = 6

def cvMat2tkImg(arr):           # Convert OpenCV image Mat to image for display
    rgb = cv.cvtColor(arr, cv.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(img)

class App(Frame):
    def __init__(self, winname='OpenCV'):       # GUI Design

        self.root = Tk()
        self.stopflag = True
        self.buffer = np.zeros((height, width, 3), dtype=np.uint8)

        global helv18
        helv18 = tkFont.Font(family='Helvetica', size=18, weight='bold')
        # print("Width",windowWidth,"Height",windowHeight)
        self.root.wm_title(winname)
        positionRight = int(self.root.winfo_screenwidth() / 2 - width / 2)
        positionDown = int(self.root.winfo_screenheight() / 2 - height / 2)
        # Positions the window in the center of the page.
        self.root.geometry("+{}+{}".format(positionRight, positionDown))
        self.root.wm_protocol("WM_DELETE_WINDOW", self.exitApp)
        Frame.__init__(self, self.root)
        self.pack(fill=BOTH, expand=1)
        # capture and display the first frame
        ret0, frame = camera.read()
        image = cvMat2tkImg(frame)
        self.panel = Label(image=image)
        self.panel.image = image
        self.panel.pack(side="top")
        # buttons
        global btnStart
        btnStart = Button(text="Start", command=self.startstop)
        btnStart['font'] = helv18
        btnStart.pack(side='right', pady = 2)
        # sliders
        global Slider1, Slider2
        Slider2 = Scale(self.root, from_=0, to=255, length= 255, orient='horizontal')
        Slider2.pack(side='right')
        Slider2.set(255)
        Slider1 = Scale(self.root, from_=0, to=255, length= 255, orient='horizontal')
        Slider1.pack(side='right')
        Slider1.set(0)
        # radio buttons
        global mode
        mode = tk.IntVar()
        mode.set(ORIGINAL)
        Radiobutton(self.root, text="Original", variable=mode, value=ORIGINAL).pack(side = 'left', pady = 4)
        Radiobutton(self.root, text="Binary", variable=mode, value=BINARY).pack(side = 'left', pady = 4)
        Radiobutton(self.root, text="Edge", variable=mode, value=EDGE).pack(side = 'left', pady = 4)
        Radiobutton(self.root, text="Line", variable=mode, value=LINE).pack(side='left', pady=4)
        Radiobutton(self.root, text="Abs Diff", variable=mode, value=ABSDIFF).pack(side='left', pady=4)
        Radiobutton(self.root, text="RGB", variable=mode, value=RGB).pack(side='left', pady=4)
        Radiobutton(self.root, text="HSV", variable=mode, value=HSV).pack(side='left', pady=4)
        # threading
        self.stopevent = threading.Event()
        self.thread = threading.Thread(target=self.capture, args=())
        self.thread.start()

    def capture(self):
        while not self.stopevent.is_set():
            if not self.stopflag:
                ret0, frame = camera.read()
                if mode.get() == BINARY:
                    if Slider1.get() > 0 and Slider1.get() < 255:
                        frame = cv.inRange(frame, (Slider1.get(), Slider1.get(), Slider1.get()), (Slider2.get(), Slider2.get(), Slider2.get()))
                elif mode.get() == EDGE:
                    frame = cv.Canny(frame, Slider1.get(), Slider2.get())
                elif mode.get() == LINE:
                    gray = cv.Canny(frame, Slider1.get(), Slider2.get())
                    lines = cv.HoughLinesP(gray, 1, np.pi/180, 100, minLineLength=10, maxLineGap=30)
                    if lines is None: continue
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                elif mode.get() == ABSDIFF:
                    temp = frame
                    frame = cv.absdiff(frame, self.buffer)
                    self.buffer = temp
                elif mode.get() == RGB:
                    half = cv.resize(frame, (int(width/2), int(height/2)))
                    b,g,r = cv.split(half)
                    top = cv.hconcat([half, cv.merge((r, r, r))])
                    bottom = cv.hconcat([cv.merge((g, g, g)), cv.merge((b, b, b))])
                    frame = cv.vconcat([top, bottom])

                image = cvMat2tkImg(frame)
                self.panel.configure(image=image)
                self.panel.image = image
                videoout.write(frame)

    def startstop(self):        #toggle flag to start and stop
        if btnStart.config('text')[-1] == 'Start':
            btnStart.config(text='Stop')
        else:
            btnStart.config(text='Start')
        self.stopflag = not self.stopflag

    def run(self):              #run main loop
        self.root.mainloop()

    def exitApp(self):          #exit loop
        self.stopevent.set()
        self.root.quit()


app = App()
app.run()
#release the camera
camera.release()
cv.destroyAllWindows()
