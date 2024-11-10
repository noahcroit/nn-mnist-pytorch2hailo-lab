import cv2
import time
import threading
import queue
import numpy as np
import onnxruntime

def preprocess(frame_in):
    # Thresholding to remove background
    tmp = cv2.cvtColor(frame_in, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(tmp, 100, 255, cv2.THRESH_BINARY)
    frame_in = cv2.bitwise_or(frame_in, frame_in, mask=mask)

    # Increase the contrast of input image
    ycrcb_image = cv2.cvtColor(frame_in, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)
    y_channel_stretched = cv2.normalize(y_channel, None, 0, 255, cv2.NORM_MINMAX)
    contrast_stretched_ycrcb = cv2.merge([y_channel_stretched, cr_channel, cb_channel])
    frame_in = cv2.cvtColor(contrast_stretched_ycrcb, cv2.COLOR_YCrCb2BGR)

    # Resize to 28x28, Invert and Normalize the pixel values
    frame_in = cv2.bitwise_not(frame_in)
    frame_out = cv2.resize(frame_in, (28, 28), interpolation=cv2.INTER_AREA)
    frame_out = frame_out / 256
    return frame_out

def task_inputframe():
    while stream.isOpened():
        # read frame
        ret, frame = stream.read()
        if not ret:
            break

        # put frame to queue
        q.put(frame)
    stream.release()

def task_processframe():
    # Load the ONNX model
    session = onnxruntime.InferenceSession('model_mnist.onnx')

    while stream.isOpened():
        # drop old frames and read the recent frame
        frame_cnt = q.qsize()
        if frame_cnt != 0:
            for i in range(frame_cnt):
                frame = q.get()

            # preprocess
            frame_out = preprocess(frame)
            # Add extra dimension to numpy image due to onnx model (input size = 1x28x28)
            # and select only 1 color channel
            data_input = frame_out[:,:,0]
            data_input = data_input.astype(np.float32)
            data_input = np.expand_dims(data_input, axis=0)

            # run the MNIST ONNX model
            predict = session.run(None, {'input': data_input})
            indices_predict = np.argmax(predict)
            print("predicted number=", int(indices_predict))

            # output / display
            font = cv2.FONT_HERSHEY_SIMPLEX
            h, w, _ = frame.shape
            c= (0, 255, 255)
            cv2.putText(frame, str(indices_predict), (0, h - 40), font, 10, c, 4, cv2.LINE_AA)
            cv2.imshow("input", frame)
            cv2.imshow("output", frame_out)

            #check keyboard for exit program
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
    stream.release()

source = "/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_5B3499FF-video-index0"
#source = 0
stream = cv2.VideoCapture(source)

q = queue.Queue()
lock = threading.Lock()

# create threads and start
t1 = threading.Thread(target=task_inputframe)
t2 = threading.Thread(target=task_processframe)

t1.start()
t2.start()
t1.join()
t2.join()
print("capture is over")

