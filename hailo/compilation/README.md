HAILO compilation process

1. Translation
    To convert .onnx file to .har file (see onnx2har.py).
    Use "Parse" command for hailo commandline.

2. Quantization
    To quantize/optimize the model to fit in 32-bit float system used in Hailo HW.
    Requirements
    - .har file (not be quantized yet)
    - .npy file as calibration file. It contains the training data and it should contain all the data as much as possible.
    the calibration should be normalized before being used in optimize command. 
    otherwise, you have to add normalization in the workflow.

    Example hailo command
    $ hailo optimize --hw-arch hailo8 --calib-set-path calib_set.npy output/model_iris.har

3. Compilation
    To compile .har file and generate binary .hef for hailo-rt, and use it in inference/deployment
    (see compile.py for example)
    

Python version for DFC
The official DFC v3.x wheels distributed for Hailo-8 target Python 3.10 because of strict underlying package dependencies (specifically, its heavy reliance on TensorFlow 2.12 and specific NumPy v1.23 versions which do not natively support Python 3.12). So, python10 can obly be used for Hailo-8.

Only for Ubuntu 24.04 (no python10 in apt)
# Install the tool required to manage PPA repositories
sudo apt install software-properties-common -y
# Add the deadsnakes PPA to your system
sudo add-apt-repository ppa:deadsnakes/ppa -y
# Update your local package index
sudo apt update

# Install Python 3.10 and its virtual environment package
sudo apt install python3.10 python3.10-venv -y
virtualenv --python=python3.10 hailo_dfc_env
source hailo_dfc_env/bin/activate

Tutorial on DFC:
https://hailo.ai/developer-zone/documentation/dataflow-compiler-v5-4-0/?sp_referrer=tutorials_notebooks/notebooks/DFC_0_Introduction.html


YOLOv5 (yolov5n.onnx) -> yolov5n.hef
Files: yolov5n_config.json, yolov5n.alls, make_calib_yolo.py (same onnx2har.py / optimize.py / compile.py)

0. ONNX export (Ultralytics yolov5, static 640x640, float32):
    $ python export.py --weights yolov5n.pt --include onnx --imgsz 640 --opset 17
   Do NOT use --dynamic or --half. Copy it to ../models/yolov5n.onnx

1. Parse
    $ python onnx2har.py -c yolov5n_config.json
   The end nodes must be the 3 last Conv layers of the Detect head ("name_end" in the config),
   NOT "output0". After these Conv layers YOLOv5 decodes the boxes with Reshape / Add / Mul on 5D
   tensors, which the Hailo cannot run, so parsing up to "output0" fails with
   "UnsupportedShuffleLayerError in op /model.24/Reshape_1".
   Check the names in Netron if your model is different (yolov5s/n/m all use /model.24/m.0..2/Conv).
   Run all the commands from this compilation folder (the paths in the config are relative to it).

2. Calibration set: 1024 photos recommended (about 64 is the minimum to run), e.g. COCO val2017
   (http://images.cocodataset.org/zips/val2017.zip, unzip into calib_images/) or pictures from the Pi camera
    $ python make_calib_yolo.py -i calib_images -o ../calib_set_yolo.npy
   1024 images -> about 1.2 GB .npy file (uint8).
   Letterboxed 640x640 RGB, values 0..255 (not normalized: yolov5n.alls adds the normalization).

3. Optimize (quantization) with the model script yolov5n.alls ("model_script" in the config)
    $ python optimize.py -c yolov5n_config.json
   yolov5n.alls:
    - normalization 0..255 -> 0..1 on the chip, so the Pi sends the uint8 image
    - change_output_activation(sigmoid): the sigmoid of the Detect head (it was cut off with the end nodes)
    - nms_postprocess(meta_arch=yolov5, engine=cpu): box decoding + NMS added to the HEF, done by HailoRT
      engine=cpu is needed: engine=nn_core does only the score threshold, no IoU -> duplicate boxes
      Without an NMS config file it uses the default YOLOv5 anchors and 80 classes: COCO models only.
    - model_optimization_flavor(optimization_level=1): without a GPU the DFC would drop to level 0.
      Level 1 adds bias correction: about 3 minutes on the CPU, and on 50 COCO images the quantized
      model found 85% of the float model detections (80% with level 0). Level 2+ needs a GPU.

4. Compile (a few minutes)
    $ python compile.py -c yolov5n_config.json
   -> output/yolov5n.hef. Copy it to the Pi and run:
    $ python3 inference_yolov5_hailo.py -e models/yolov5n.hef --mode camera
   Compare with: python3 inference_yolov5_onnx.py -e models/yolov5n.onnx --mode camera
