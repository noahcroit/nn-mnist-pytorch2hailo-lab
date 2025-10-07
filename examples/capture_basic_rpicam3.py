import cv2
from picamera2 import Picamera2
import numpy as np
import time



def process():
    time.sleep(0.2)



picam2 = Picamera2()
config = picam2.create_still_configuration(
        main={"size": (1280, 640), "format": 'RGB888'},  # Custom resolution for the final image
    lores={"size": (640, 480)},   # Low-resolution stream for preview/display
    display="lores"               # Tell it to use the 'lores' stream for the default preview window
)
picam2.configure(config)
picam2.start()

print("start capture")
while True:
    # read frame
    frame = picam2.capture_array()
    if not frame.any():
        break

    # some blocking process
    process()

    # output / display
    cv2.imshow("output", frame)

    # check keyboard for exit program
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break

picam2.stop()
picam2.close()
