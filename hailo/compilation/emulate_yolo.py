"""
Test the optimized YOLOv5 HAR on the PC with the Hailo emulator, before compiling / deploying.

    python emulate_yolo.py -c yolov5n_config.json -i ../img_yolo

Runs every photo twice:
    SDK_FP_OPTIMIZED : float model + model script (normalization, NMS) -> checks the parsing and the .alls
    SDK_QUANTIZED    : 8-bit emulation of the chip                     -> what the Hailo-8 will output
If FP_OPTIMIZED is good but QUANTIZED is bad, the quantization (calibration set) is the problem.
The pictures with the boxes are saved to output/emulator/.
"""
import argparse
import glob
import json
import os

import numpy as np
from hailo_sdk_client import ClientRunner, InferenceContext
from PIL import Image, ImageDraw

from make_calib_yolo import letterbox

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def get_detections(nms_out, conf_threshold, input_size):
    """Emulator NMS output of one image: classes x 5 x max_boxes, rows [y0, x0, y1, x1, score] 0..1.
    Returns a list of (class_id, score, [x0, y0, x1, y1] in pixels of the model input)."""
    detections = []
    for class_id in range(nms_out.shape[0]):
        for y0, x0, y1, x1, score in nms_out[class_id].T:
            if score >= conf_threshold:
                box = [x0 * input_size, y0 * input_size, x1 * input_size, y1 * input_size]
                detections.append((class_id, float(score), box))
    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv5 HAR test with the Hailo emulator")
    parser.add_argument("-c", "--cfg", help="configuration file as .json file", default="yolov5n_config.json")
    parser.add_argument("-i", "--images", help="folder with .jpg / .png test photos", default="../img_yolo")
    parser.add_argument("--conf", type=float, default=0.25, help="minimum score of a detection")
    parser.add_argument("--size", type=int, default=640, help="model input size")
    args = parser.parse_args()
    with open(args.cfg) as f:
        data = json.load(f)

    files = sorted(f for f in glob.glob(os.path.join(args.images, "*"))
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        raise SystemExit(f"no images found in {args.images}")
    # same input as on the Pi: letterboxed RGB, 0..255 (the normalization is in the model)
    images = np.stack([letterbox(Image.open(p).convert("RGB"), args.size, args.size) for p in files])

    runner = ClientRunner(har=data["output_path"] + data["quantized_har_name"])
    out_dir = os.path.join(data["output_path"], "emulator")
    os.makedirs(out_dir, exist_ok=True)

    for context_name in ["SDK_FP_OPTIMIZED", "SDK_QUANTIZED"]:
        with runner.infer_context(getattr(InferenceContext, context_name)) as ctx:
            results = runner.infer(ctx, images.astype(np.float32))
        print(f"\n{context_name}:")
        for path, img, nms_out in zip(files, images, results):
            detections = get_detections(np.asarray(nms_out), args.conf, args.size)
            names = [f"{COCO_LABELS[c]} {s:.2f}" for c, s, _ in detections]
            print(f"  {os.path.basename(path)}: {len(detections)} objects: {', '.join(names)}")

            # draw the boxes on the model input image and save it
            picture = Image.fromarray(img)
            draw = ImageDraw.Draw(picture)
            for class_id, score, box in detections:
                draw.rectangle(box, outline=(0, 255, 0), width=2)
                draw.text((box[0] + 3, box[1] + 2), f"{COCO_LABELS[class_id]} {score:.2f}", fill=(0, 255, 0))
            name = os.path.splitext(os.path.basename(path))[0]
            picture.save(os.path.join(out_dir, f"{name}_{context_name}.png"))
    print("\nimages with boxes saved in", out_dir)
