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
