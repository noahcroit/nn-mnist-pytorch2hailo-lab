import cv2
from picamera2 import Picamera2
import numpy as np
import time
import threading
import queue



still_capture=False



def process():
    time.sleep(0.2)



def task_inputframe():
    global still_capture
    still_capture=True
    picam2.start()
    while still_capture:
        # read frame
        frame = picam2.capture_array()
        if not frame.any():
            break

        # put frame to queue
        q.put(frame)
        
    picam2.stop()
    picam2.close()
    still_capture=False



def task_outputframe():
    global still_capture
    time.sleep(0.5)
    while still_capture:
        # drop old frames and read the recent frame
        frame_cnt = q.qsize()
        print("num of frame in Q=", frame_cnt)
        if frame_cnt != 0:
            for i in range(frame_cnt):
                frame = q.get()

            # some process
            process()

            # output / display
            cv2.imshow("output", frame)
        
        # check keyboard for exit program
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            still_capture=False



q = queue.Queue()
lock = threading.Lock()
picam2 = Picamera2()
config = picam2.create_still_configuration(
        main={"size": (1280, 720), "format": 'RGB888'},  # Custom resolution for the final image
    lores={"size": (640, 480)},   # Low-resolution stream for preview/display
    display="lores"               # Tell it to use the 'lores' stream for the default preview window
)
picam2.configure(config)

t1 = threading.Thread(target=task_inputframe)
t2 = threading.Thread(target=task_outputframe)

t1.start()
t2.start()
t1.join()
t2.join()
print("capture is over")

