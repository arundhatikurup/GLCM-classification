import numpy as np
import cv2

# Create a black image
#img = np.zeros((512,512,3), np.uint8)
img=cv2.imread("grass-sky.jpg",1)
#612 x 407

# Write some Text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'Hello World!',(10,10), font, 0.8,(0,255,0),2)

#Display the image
cv2.imshow("img",img)

cv2.waitKey(0)
