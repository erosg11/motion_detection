import cv2
import numpy as np

cam = cv2.VideoCapture(0)

_, background = cam.read()

background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

DEBUG = False

alpha = 0.3
theta = 0.1
kernel_clear = np.ones((8, 8))
kernel_join = np.ones((75, 75))
kernel_contour = np.ones((100, 100))
min_area = 500
memory_size = 4
cell = []

while cam.isOpened():
    ret, curr_img = cam.read()
    if not ret:
        break
    gray = cv2.cvtColor(curr_img.copy(), cv2.COLOR_BGR2GRAY)
    background = (alpha * background + (1 - alpha) * gray).astype(np.uint8)
    diffImg = np.absolute(gray - background)
    cleared = cv2.dilate(cv2.erode(diffImg, kernel_clear), kernel_clear)
    joined = cv2.erode(cv2.dilate(cleared, kernel_join), kernel_join)
    _, thresh = cv2.threshold(joined, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    canny_output = cv2.Canny(thresh, 128, 255)
    _, contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull_list = [cv2.convexHull(contour) for contour in contours]
    pre_contours = np.zeros_like(curr_img)
    cv2.drawContours(pre_contours, hull_list, -1, (255, 255, 255), -1)
    filled = cv2.cvtColor(pre_contours, cv2.COLOR_BGR2GRAY)
    merged_contours = cv2.erode(cv2.dilate(filled, kernel_contour), kernel_contour)
    _, thresh = cv2.threshold(merged_contours, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull_list = [cv2.convexHull(contour) for contour in contours]
    hull_list = [hull for hull in hull_list if cv2.contourArea(hull) > min_area]
    mask = np.zeros_like(curr_img)
    cv2.drawContours(mask, hull_list, -1, (255, 255, 255), -1)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cell = cell[-memory_size:] + [mask.astype(np.bool)]
    mask = np.any(cell, axis=0).astype(np.uint8)
    maskBGR = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
    cv2.imshow("Mask", maskBGR*255)
    colored2 = curr_img
    result = cv2.bitwise_and(colored2, colored2, mask=mask)
    if DEBUG:
        for i, hull in enumerate(hull_list):
            m = cv2.moments(hull)
            i = cv2.contourArea(hull)
            try:
                cs = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
            except ZeroDivisionError:
                cs = (0, 0)
            cv2.putText(result, str(i), cs, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Result", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
