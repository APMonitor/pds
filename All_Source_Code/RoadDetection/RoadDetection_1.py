elif mode.get() == HSV:
    half = cv.resize(frame, (int(width/2), int(height/2)))
    hsv = cv.cvtColor(half, cv.COLOR_BGR2HSV)
    h,s,v = cv.split(hsv)
    if Slider1.get() > 0 and Slider1.get() < 255:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        s = cv.inRange(s, Slider1.get(), Slider2.get())
        s = cv.morphologyEx(s, cv.MORPH_OPEN, kernel)
        v = cv.inRange(v, Slider1.get(), Slider2.get())
        v = cv.morphologyEx(v, cv.MORPH_OPEN, kernel)
    top = cv.hconcat([half, cv.merge((h, h, h))])
    bottom = cv.hconcat([cv.merge((s, s, s)), cv.merge((v, v, v))])
    frame = cv.vconcat([top, bottom])