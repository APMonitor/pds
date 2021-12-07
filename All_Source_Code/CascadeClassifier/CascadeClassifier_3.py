faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,
                                     minNeighbors=4,\
                                     minSize=(30,30),\
                                     maxSize=(200,200))