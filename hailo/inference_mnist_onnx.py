"""
inference_mnist_onnx.py

Run the MNIST digit classifier (DIY ANN) with ONNX Runtime on the Raspberry Pi 5 CPU.

    python3 inference_mnist_onnx.py --mode image                      # test images img/0.jpg ... 9.jpg
    python3 inference_mnist_onnx.py --mode camera                     # Pi Camera Module 3, live FPS
    python3 inference_mnist_onnx.py --mode camera --camera usb        # USB webcam
    python3 inference_mnist_onnx.py --mode camera -r                  # camera mounted 90 deg counter clockwise

Press 'q' in the video window or Ctrl+C to quit.
Compare with inference_mnist_hailo.py (same program on the Hailo-8 NPU):
the "inference ... ms" value is the speed of the model itself. The FPS is the speed of the whole loop
and is limited by the camera (30 FPS) and the display, so it is about the same for both.
"""
import argparse
import os
import queue
import threading
import time

import cv2
import numpy as np
import onnxruntime

# folder of this script: the default "img" folder is next to it, wherever the script is started from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def frame_preprocess(frame_in):
    """Camera frame (BGR) -> 28x28 float32 image in 0..1, white digit on black like MNIST."""
    gray = cv2.cvtColor(frame_in, cv2.COLOR_BGR2GRAY)

    # Crop only ROI (centered square)
    h, w = gray.shape
    size = min(h, w)
    x0, y0 = (w - size) // 2, (h - size) // 2
    gray = gray[y0:y0 + size, x0:x0 + size]

    # Thresholding to remove background: dark ink -> 0, keep the bright paper as-is
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Increase the contrast, then invert: black ink on white paper -> white digit on black
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.bitwise_not(gray)

    # Resize to 28x28 as the input size of the ANN model, normalize to 0..1
    frame_out = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    return frame_out.astype(np.float32) / 255.0


def run_inference(session, x):
    """Run the model on one 28x28 image, return the predicted digit."""
    model_input = session.get_inputs()[0]
    # reshape the 28x28 image to the model input shape, e.g. (1, 28, 28) or (1, 784)
    # (a dynamic dimension such as "batch" is replaced by 1)
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in model_input.shape]
    x = x.reshape(shape)

    output = session.run(None, {model_input.name: x})
    return int(np.argmax(output[0]))


def put_latest(q, item):
    """Put item into a size-1 queue, dropping the old frame if present."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put_nowait(item)


def put_frame_to_queue(grab_frame, q, stop_event):
    """Producer thread: read frames from the camera and keep only the latest one in the queue."""
    try:
        while not stop_event.is_set():
            frame = grab_frame()
            if frame is None:
                print("camera returned no frame")
                break
            put_latest(q, frame)
    except Exception as e:
        if not stop_event.is_set():   # errors while closing the camera are expected
            print("camera error:", e)
    finally:
        stop_event.set()


def open_camera(camera, usb_device, width, height):
    """Returns (grab_frame, close) functions for the Pi Camera or a USB webcam."""
    if camera == "picam":
        from libcamera import controls
        from picamera2 import Picamera2

        picam2 = Picamera2()
        # "RGB888" in Picamera2 is stored in B,G,R byte order -> already what OpenCV expects
        config = picam2.create_video_configuration(main={"size": (width, height), "format": "RGB888"})
        picam2.configure(config)
        # Camera Module 3 has autofocus, but it is off by default (fixed focus far away):
        # enable continuous autofocus so a digit held close to the camera is sharp
        if "AfMode" in picam2.camera_controls:
            picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        picam2.start()
        return (lambda: picam2.capture_array("main")), (lambda: (picam2.stop(), picam2.close()))

    # on the Pi 5 the camera index is not always 0, a /dev/v4l/by-id/... path is safer
    stream = cv2.VideoCapture(int(usb_device) if usb_device.isdigit() else usb_device)
    if not stream.isOpened():
        stream.release()
        raise RuntimeError(f"cannot open the USB camera {usb_device}")
    stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def grab_frame():
        ret, frame = stream.read()
        return frame if ret else None

    return grab_frame, stream.release


def mode_image(infer, img_dir):
    """Inference with local images img/0.jpg ... 9.jpg
    infer(x) runs the model on one 28x28 image and returns the predicted digit."""
    for number in range(10):
        # read image
        img_path = os.path.join(img_dir, f"{number}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            print("cannot read", img_path)
            continue

        # preprocess image and run inference
        x = frame_preprocess(img)
        t_start = time.perf_counter()
        predict = infer(x)
        t_infer = time.perf_counter() - t_start
        print(f"{img_path}: predicted number = {predict}  (inference {t_infer * 1000:.2f} ms)")

        # output / display
        cv2.putText(img, str(predict), (0, img.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX,
                    10, (0, 255, 255), 4, cv2.LINE_AA)
        cv2.imshow("input number", img)
        cv2.imshow("model input", cv2.resize(x, (280, 280), interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(1500) & 0xFF == ord("q"):
            break


def mode_camera(infer, camera, usb_device, width, height, rotate=False):
    """Inference with the live camera, show the FPS until 'q' or Ctrl+C.
    infer(x) runs the model on one 28x28 image and returns the predicted digit.
    rotate=True turns the frame 90 degrees clockwise, for a camera mounted 90 degrees counter clockwise."""
    grab_frame, close_camera = open_camera(camera, usb_device, width, height)
    q = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    t1 = threading.Thread(target=put_frame_to_queue, args=(grab_frame, q, stop_event), daemon=True)
    t1.start()

    fps = 0.0
    t_infer_avg = 0.0
    t_previous = time.perf_counter()
    t_print = t_previous
    try:
        while not stop_event.is_set():
            # wait for the most recent frame
            try:
                frame = q.get(timeout=0.5)
            except queue.Empty:
                continue

            # rotate back the frame of a camera mounted 90 degrees counter clockwise
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # preprocess image and run inference
            x = frame_preprocess(frame)
            t_start = time.perf_counter()
            predict = infer(x)
            t_infer = time.perf_counter() - t_start

            # FPS measurement (wall clock, smoothed)
            t_current = time.perf_counter()
            fps = 0.9 * fps + 0.1 / (t_current - t_previous)
            t_infer_avg = 0.9 * t_infer_avg + 0.1 * t_infer
            t_previous = t_current
            if t_current - t_print >= 1.0:
                t_print = t_current
                print(f"FPS = {fps:5.1f} | inference {t_infer_avg * 1000:6.2f} ms | predicted number = {predict}")

            # output / display
            font = cv2.FONT_HERSHEY_SIMPLEX
            h, w, _ = frame.shape
            txcolor = (0, 255, 255)
            cv2.putText(frame, f"ONNX (CPU)  FPS={fps:.1f}", (10, 40), font, 1.2, txcolor, 3, cv2.LINE_AA)
            cv2.putText(frame, f"inference {t_infer_avg * 1000:.2f} ms", (10, 85), font, 1.2, txcolor, 3, cv2.LINE_AA)
            cv2.putText(frame, str(predict), (0, h - 40), font, 10, txcolor, 4, cv2.LINE_AA)
            size = min(h, w)
            cv2.rectangle(frame, ((w - size) // 2, (h - size) // 2), ((w + size) // 2, (h + size) // 2), txcolor, 2)
            cv2.imshow("input", frame)
            cv2.imshow("model input", cv2.resize(x, (280, 280), interpolation=cv2.INTER_NEAREST))

            # check keyboard for exit program
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stop_event.set()
        t1.join(timeout=2.0)
        close_camera()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST inference with ONNX Runtime (CPU)")
    parser.add_argument("-e", "--model", help="ONNX model (.onnx)", default="models/model_mnist.onnx")
    parser.add_argument("-m", "--mode", help="test with local images or the live camera",
                        choices=["image", "camera"], default="image")
    parser.add_argument("-i", "--images", help="folder with 0.jpg ... 9.jpg (image mode)",
                        default=os.path.join(SCRIPT_DIR, "img"))
    parser.add_argument("-c", "--camera", help="Pi Camera Module 3 or USB webcam (camera mode)",
                        choices=["picam", "usb"], default="picam")
    parser.add_argument("-u", "--usb-device", help="USB webcam: index or /dev/v4l/by-id/... path",
                        default="0")
    parser.add_argument("-r", "--rotate", action="store_true",
                        help="rotate the camera frame 90 deg clockwise (camera mounted 90 deg counter clockwise)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    # Load the ONNX model
    session = onnxruntime.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    print("input :", model_input.name, model_input.shape)
    print("output:", session.get_outputs()[0].name, session.get_outputs()[0].shape)

    def infer(x):
        return run_inference(session, x)

    try:
        if args.mode == "image":
            mode_image(infer, args.images)
        else:
            mode_camera(infer, args.camera, args.usb_device, args.width, args.height, args.rotate)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("inference is over")
