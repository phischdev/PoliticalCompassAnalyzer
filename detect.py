import argparse
import numpy as np
from skimage import measure

import cv2
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("image")

args = parser.parse_args()

debug = []

image_path = args.image
image = cv2.imread(image_path)
debug.append(("input", image))

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
debug.append(("gray", gray))
ret, thresh = cv2.threshold(gray, 79, 255, cv2.THRESH_BINARY_INV)
# debug.append(("thresh", thresh))

# thresh = thresh < 50
# crop = thresh[np.ix_(thresh.any(1),thresh.any(0))]

_, crop_contour, _ = cv2.findContours(
    thresh,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE)

# img_contour = image.copy()
# cv2.drawContours(img_contour, crop_contour, 0 , (0,255,0), 2)
debug.append(("thresh", thresh))

cnt = crop_contour[0]
x, y,w, h = cv2.boundingRect(cnt)
crop = thresh[y:y+h, x:x+w]
debug.append(("crop", crop))

kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(crop,kernel,iterations = 1)
debug.append(("erosion", erosion))

_, circles,_ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

fin = image.copy()
fin = fin[y:y+h, x:x+w]

for circle in circles:

    ((cX, cY), radius) = cv2.minEnclosingCircle(circle)

    cv2.circle(fin, (int(cX), int(cY)), int(radius), (0, 0, 255), 3)

    (w, h) = erosion.shape
    (relX, relY) = (cX - w/2, cY - h/2)
    (polit_x, polit_y) = (round(relX*10 / w, 1), round(relY * 10 / h, 1))
    cv2.putText(fin, str((polit_x, polit_y)), (int(cX), int(cY)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

debug.append(("final", fin))


for i, (label, img) in enumerate(debug):
    plt.subplot(3,2,i+1), plt.imshow(img, 'gray')
    plt.xticks([]), plt.yticks([])
    plt.title(label)

plt.show()

#cv2.waitKey(0)


