import cv2
import numpy as np
import time

filename = './Project/Tho/Linux/data/lena.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
start = time.time()
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
stop = time.time()
print("Execution time: %0.2f" % (stop - start))

# result is dilated for marking the corners, not important
dst = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst > 0.01*dst.max()] = [0, 0, 255]

cv2.imshow('dst', img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
