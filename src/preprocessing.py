import cv2
import numpy as np

def clean_image(img):
    # 1. Safe Crop (Remove artifacts at the bottom)
    h, w = img.shape[:2]
    # Cut bottom 10% where artifacts often appear
    img = img[0:int(h*0.90), :] 

    # 2. Inpaint Date Stamp (Remove orange text)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Define orange color range (adjust as needed)
    lower = np.array([5, 150, 150])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Dilate mask to cover text edges and reduce noise
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=2)

    # Inpaint if mask is not empty
    if np.sum(mask) > 0:
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    return img