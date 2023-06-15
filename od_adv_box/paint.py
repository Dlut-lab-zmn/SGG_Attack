import cv2
import numpy as np
import random
font = cv2.FONT_HERSHEY_SIMPLEX
img = np.zeros((100,100,3))
color = tuple([random.randint(0,255) for _ in range(3)])
name = 'pole'
img = cv2.putText(img, '{}'.format(name), (20, 20), font, 0.7,(0, 255, 0), 2)
cv2.imwrite('1.png',img)
