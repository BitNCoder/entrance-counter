import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

_, base_image = cap.read()


# Wonderful blur shape
BLUR_SHAPE = (13, 13)
KERNEL = np.ones((5,5), np.uint8)


while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame...")
        break
    

    # Show the new frame (live camera feed)
    cv.imshow("image", frame)

    # Show the "base_image" (the image the feed is compared with)
    cv.imshow("base_image", base_image)

    # Convert both to gray-scale
    gframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gbase = cv.cvtColor(base_image, cv.COLOR_BGR2GRAY)

    # apply a blur to both to reduce noise and computing power
    bgframe = cv.GaussianBlur(gframe, BLUR_SHAPE, 0)
    bgbase = cv.GaussianBlur(gbase, BLUR_SHAPE, 0)

    # This is a numpy difference, which wraps values with great enough differences (integer overflow)
    # Ex. (unsigned 8-bit integers) 8 - 10 = 253 instead of 0
    # This allows for differences between images to be preserved while similar values can be removed through errosion and thresholding
    difference = bgframe - bgbase
    cv.imshow("difference", difference)

    # Threshold the difference to make a "mask" (removes low values from the image)
    ret, tdifference = cv.threshold(difference, 60, 255, cv.THRESH_BINARY)
    cv.imshow("tdifference", tdifference)

    # Remove high values of noise
    errode = cv.erode(tdifference, KERNEL, iterations=5)
    cv.imshow("errode", errode)

    # Fill in shapes and attempt to restore removed information
    dialate = cv.dilate(errode, KERNEL, iterations=10)
    cv.imshow("dialate", dialate)

    # Additional errode to clean up edges
    erode2 = cv.erode(dialate, KERNEL, iterations=3)
    cv.imshow("erode2", erode2)

    # use it as a mask :)
    # This shows what is seen as different in the current frame relative to the base_image
    masked_frame = cv.bitwise_and(frame, frame, mask=erode2)
    cv.imshow("pseudo greenscreen", masked_frame)

    # Grab the "contours" the image --> hopefully grab people in the image
    contours, hist = cv.findContours(erode2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = list(filter(lambda x: x.size > 30, contours)) # Remove small contours (likely to be noise)
    dc = cv.drawContours(masked_frame, contours, -1, (0, 255, 0), 3) # Draw the contours to show what is being seen as a contour

    # Draw the centroids of each of the contours --> (this may be how people and their motion is tracked)
    for c in contours:
        moment = cv.moments(c)
        cx = int(moment['m10']/moment['m00'])
        cy = int(moment['m01']/moment['m00'])
        dc = cv.circle(dc, (cx, cy), 4, (255, 0, 0), 3)
    cv.imshow("dc", dc)

    key_input = cv.waitKey(1)
    # Quit the program when 'q' is pressed
    if key_input == ord('q'):
        break
    # Grab a new base_image when 'c' is pressed
    elif key_input == ord('c'):
        base_image = frame

    #base_image = frame
for c in contours:
    print(f'{c.size=}, {c.shape=}')

cap.release()
cv.destroyAllWindows()