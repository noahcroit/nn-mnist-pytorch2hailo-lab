import cv2
import time

def process():
    time.sleep(0.2)

#source = "/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_5B3499FF-video-index0"
source = 0
stream = cv2.VideoCapture(source)

print("start capture")
while stream.isOpened():
    # read frame
    ret, frame = stream.read()

    # some blocking process
    process()

    # output / display
    cv2.imshow("output", frame)

    # check keyboard for exit program
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break

stream.release()
print("capture is over")

