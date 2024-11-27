from hailo_sdk_client import ClientRunner
import argparse
import json



parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("-c", "--cfg", help="ONNX 2 HAR configuration file as .json file", default='.json')
args = parser.parse_args()
f = open(args.cfg)
data = json.load(f)

model_name = data['model_name']
hailo_model_har_name = data['output_path'] + data['har_name']
runner = ClientRunner(har=hailo_model_har_name)

# Normalization Section
# Now we will create a model script, that tells the compiler to add a normalization on the beginning
# of the model (that is why we didn't normalize the calibration set;
# Otherwise we would have to normalize it before using it)
# Batch size is 8 by default
#alls = "normalization1 = normalization([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])\n"
#runner.load_model_script(alls)

# Call Optimize to perform the optimization process
runner.optimize(data['calib_set'])
# Save the result state to a Quantized HAR file
quantized_model_har_path = data['output_path'] + data['quantized_har_name']
runner.save_har(quantized_model_har_path)
