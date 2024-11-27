from hailo_sdk_client import ClientRunner
import argparse
import json



parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("-c", "--cfg", help="ONNX 2 HAR configuration file as .json file", default='.json')
args = parser.parse_args()
f = open(args.cfg)
data = json.load(f)

quantized_har_path = data["output_path"] + data["quantized_har_name"]
runner = ClientRunner(har=quantized_har_path)
hef = runner.compile()
file_name = data["output_path"] + data["model_name"] + ".hef"
with open(file_name, "wb") as f:
    f.write(hef)
