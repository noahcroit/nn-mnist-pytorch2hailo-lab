"""
inference_yolov5_hailo.py

Run the YOLOv5 object detector (COCO, 80 classes) on the Hailo-8 NPU with HailoRT.

    python3 inference_yolov5_hailo.py --mode image                    # all images in img_yolo/
    python3 inference_yolov5_hailo.py --mode camera                   # Pi Camera Module 3, live FPS
    python3 inference_yolov5_hailo.py --mode camera --camera usb      # USB webcam
    python3 inference_yolov5_hailo.py --mode camera -r                # camera mounted 90 deg counter clockwise

Press 'q' in the video window or Ctrl+C to quit.
Compare with inference_yolov5_onnx.py (same program on the Raspberry Pi 5 CPU):
the "inference ... ms" value is the time from the model input image to the list of detected objects
(model + NMS). YOLOv5s is about 50000x more work than the MNIST ANN, so here the NPU is much faster.

Model: YOLOv5 (s or n) compiled for Hailo-8 WITH the NMS post-process in the HEF (as the Hailo Model Zoo does).
The network runs on the NPU, the box decoding + NMS is done by HailoRT (C++) before the result comes to Python.
    - precompiled: download yolov5s.hef for hailo8 from the Hailo Model Zoo, matching your HailoRT version
    - or compile:  hailomz compile yolov5s --hw-arch hailo8   (Hailo Dataflow Compiler + Model Zoo, on the PC)
    - or your own ONNX: see compilation/README.md (yolov5n_config.json + yolov5n.alls)
"""
import argparse
import glob
import os
import queue
import threading
import time

import cv2
import numpy as np
from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoSchedulingAlgorithm,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)

# folder of this script: the default "img_yolo" folder is next to it, wherever the script is started from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


def frame_preprocess(frame_in, input_h, input_w):
    """Camera frame (BGR) -> RGB image of the model input size (uint8, H x W x 3).
    Letterbox: resize keeping the aspect ratio, pad the rest with gray (as in YOLOv5 training).
    Returns the image, the scale ratio and the (pad_x, pad_y) offsets to map the boxes back."""
    h, w = frame_in.shape[:2]
    ratio = min(input_h / h, input_w / w)
    new_h, new_w = int(round(h * ratio)), int(round(w * ratio))
    img = cv2.resize(frame_in, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_y, pad_x = (input_h - new_h) // 2, (input_w - new_w) // 2
    img = cv2.copyMakeBorder(img, pad_y, input_h - new_h - pad_y, pad_x, input_w - new_w - pad_x,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, ratio, (pad_x, pad_y)


def run_inference(infer_pipeline, input_vstream_info, output_vstream_info, img, conf_threshold):
    """Run the model on one RGB image, return the detections in model input coordinates:
    boxes (N x 4: x0, y0, x1, y1), scores (N), classes (N)."""
    # HWC uint8 0..255 -> add the batch dimension (the normalization is compiled into the HEF)
    x = np.ascontiguousarray(img[np.newaxis])

    infer_results = infer_pipeline.infer({input_vstream_info.name: x})
    # NMS is already done (HEF post-process in HailoRT). Output of batch 0 = one array per class with the rows
    # [y0, x0, y1, x1, score], coordinates normalized 0..1 to the model input
    detections_per_class = infer_results[output_vstream_info.name][0]
    if len(detections_per_class) != len(COCO_LABELS):
        raise ValueError(f"expected {len(COCO_LABELS)} classes in the NMS output, got {len(detections_per_class)}: "
                         "use a YOLOv5 COCO HEF with the NMS post-process (Hailo Model Zoo yolov5s)")

    input_h, input_w = img.shape[:2]
    boxes, scores, classes = [], [], []
    for class_id, detections in enumerate(detections_per_class):
        for y0, x0, y1, x1, score in np.asarray(detections).reshape(-1, 5):
            if score >= conf_threshold:
                boxes.append([x0 * input_w, y0 * input_h, x1 * input_w, y1 * input_h])
                scores.append(score)
                classes.append(class_id)
    return (np.array(boxes, dtype=np.float32).reshape(-1, 4),
            np.array(scores, dtype=np.float32), np.array(classes, dtype=int))


def scale_boxes(boxes, ratio, pad, frame_shape):
    """Model input coordinates -> camera frame coordinates (undo the letterbox)."""
    pad_x, pad_y = pad
    boxes = (boxes - [pad_x, pad_y, pad_x, pad_y]) / ratio
    h, w = frame_shape[:2]
    return np.clip(boxes, 0, [w - 1, h - 1, w - 1, h - 1])


def draw_detections(frame, boxes, scores, classes):
    for (x0, y0, x1, y1), score, class_id in zip(boxes.astype(int), scores, classes):
        color = (int(class_id) * 67 % 256, int(class_id) * 131 % 256, 255 - int(class_id) * 29 % 256)
        label = f"{COCO_LABELS[class_id]} {score:.2f}"
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x0, max(0, y0 - th - 8)), (x0 + tw + 4, y0), color, -1)
        cv2.putText(frame, label, (x0 + 2, max(th, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 2, cv2.LINE_AA)


def describe(classes):
    """e.g. '3 objects: person x2, cup x1'"""
    names, counts = np.unique([COCO_LABELS[c] for c in classes], return_counts=True)
    return f"{len(classes)} objects" + (": " if len(classes) else "") + \
        ", ".join(f"{n} x{c}" for n, c in zip(names, counts))


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
        # enable continuous autofocus
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


def mode_image(infer, input_size, img_dir):
    """Inference with all local images (.jpg / .png) of img_dir.
    infer(img) runs the model on one model input image and returns (boxes, scores, classes)."""
    files = sorted(f for f in glob.glob(os.path.join(img_dir, "*"))
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        print("no image found in", img_dir)
    for img_path in files:
        # read image
        frame = cv2.imread(img_path)
        if frame is None:
            print("cannot read", img_path)
            continue

        # preprocess image and run inference
        img, ratio, pad = frame_preprocess(frame, *input_size)
        t_start = time.perf_counter()
        boxes, scores, classes = infer(img)
        t_infer = time.perf_counter() - t_start
        print(f"{img_path}: {describe(classes)}  (inference {t_infer * 1000:.1f} ms)")

        # output / display
        draw_detections(frame, scale_boxes(boxes, ratio, pad, frame.shape), scores, classes)
        cv2.imshow("input image", frame)
        if cv2.waitKey(2000) & 0xFF == ord("q"):
            break


def mode_camera(infer, input_size, camera, usb_device, width, height, rotate=False):
    """Inference with the live camera, show the FPS until 'q' or Ctrl+C.
    infer(img) runs the model on one model input image and returns (boxes, scores, classes).
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
            img, ratio, pad = frame_preprocess(frame, *input_size)
            t_start = time.perf_counter()
            boxes, scores, classes = infer(img)
            t_infer = time.perf_counter() - t_start

            # FPS measurement (wall clock, smoothed)
            t_current = time.perf_counter()
            fps = 0.9 * fps + 0.1 / (t_current - t_previous)
            t_infer_avg = 0.9 * t_infer_avg + 0.1 * t_infer
            t_previous = t_current
            if t_current - t_print >= 1.0:
                t_print = t_current
                print(f"FPS = {fps:5.1f} | inference {t_infer_avg * 1000:6.1f} ms (max {1 / t_infer_avg:5.1f} FPS)"
                      f" | {describe(classes)}")

            # output / display
            draw_detections(frame, scale_boxes(boxes, ratio, pad, frame.shape), scores, classes)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txcolor = (0, 255, 255)
            cv2.putText(frame, f"Hailo-8 (NPU)  FPS={fps:.1f}", (10, 40), font, 1.2, txcolor, 3, cv2.LINE_AA)
            # max FPS = what the model alone could reach (the loop FPS is limited by the camera)
            cv2.putText(frame, f"inference {t_infer_avg * 1000:.1f} ms (max {1 / t_infer_avg:.0f} FPS)", (10, 85),
                        font, 1.2, txcolor, 3, cv2.LINE_AA)
            cv2.imshow("input", frame)

            # check keyboard for exit program
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stop_event.set()
        t1.join(timeout=2.0)
        close_camera()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv5 inference on the Hailo-8 NPU")
    parser.add_argument("-e", "--model", help="Hailo Executable File (.hef)", default="models/yolov5s.hef")
    parser.add_argument("-m", "--mode", help="test with local images or the live camera",
                        choices=["image", "camera"], default="image")
    parser.add_argument("-i", "--images", help="folder with .jpg / .png images (image mode)",
                        default=os.path.join(SCRIPT_DIR, "img_yolo"))
    parser.add_argument("-c", "--camera", help="Pi Camera Module 3 or USB webcam (camera mode)",
                        choices=["picam", "usb"], default="picam")
    parser.add_argument("-u", "--usb-device", help="USB webcam: index or /dev/v4l/by-id/... path",
                        default="0")
    parser.add_argument("-r", "--rotate", action="store_true",
                        help="rotate the camera frame 90 deg clockwise (camera mounted 90 deg counter clockwise)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--conf", type=float, default=0.25,
                        help="minimum score of a detection (the NMS overlap threshold is fixed in the HEF)")
    args = parser.parse_args()

    # Disable the model scheduler: the network group is activated manually below.
    # Only one program can use the Hailo-8 at a time.
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.NONE

    # The "with" statements release the NPU when the program ends, also after Ctrl+C or an error
    with VDevice(params=params) as target:
        # Loading compiled HEF to device
        hef = HEF(args.model)
        configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        # Create input and output virtual streams params
        # UINT8 input   : the 0..255 pixels are sent as-is (normalization is inside the HEF)
        # FLOAT32 output: the NMS boxes and scores as float
        input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

        input_vstream_info = hef.get_input_vstream_infos()[0]
        output_vstream_info = hef.get_output_vstream_infos()[0]
        print("input :", input_vstream_info.name, input_vstream_info.shape)
        print("output:", output_vstream_info.name, output_vstream_info.shape)
        input_size = tuple(input_vstream_info.shape[0:2])   # (H, W, 3) -> (H, W)

        # Open the inference pipeline and activate the network group ONCE, for all frames
        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline, \
                network_group.activate(network_group_params):

            def infer(img):
                return run_inference(infer_pipeline, input_vstream_info, output_vstream_info, img, args.conf)

            # warm-up: the first inference is slower (memory allocation), do not count it
            infer(np.zeros(input_size + (3,), np.uint8))

            try:
                if args.mode == "image":
                    mode_image(infer, input_size, args.images)
                else:
                    mode_camera(infer, input_size, args.camera, args.usb_device, args.width, args.height,
                                args.rotate)
            except KeyboardInterrupt:
                pass
            finally:
                cv2.destroyAllWindows()
                print("inference is over")
