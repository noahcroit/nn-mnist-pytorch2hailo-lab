import cv2
import time
import threading
import queue

def process():
    time.sleep(0.2)

source = "/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_5B3499FF-video-index0"
source = 0
stream = cv2.VideoCapture(source)
q = queue.Queue()
lock = threading.Lock()

def task_inputframe():
    while stream.isOpened():
        # read frame
        ret, frame = stream.read()
        if not ret:
            break

        # put frame to queue
        q.put(frame)
    stream.release()

def task_outputframe():
    while stream.isOpened():
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

            #check keyboard for exit program
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
    stream.release()

t1 = threading.Thread(target=task_inputframe)
t2 = threading.Thread(target=task_outputframe)

t1.start()
t2.start()
t1.join()
t2.join()
print("capture is over")

