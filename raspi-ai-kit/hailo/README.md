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
