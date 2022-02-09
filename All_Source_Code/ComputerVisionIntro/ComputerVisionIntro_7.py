width  = 300      # new width
h,w,c  = im.shape # get image size
scale  = width/w  # scaling factor
height = int(h * scale)
dim    = (width, height)
im3    = cv2.resize(im,dim)