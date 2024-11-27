from IPython.display import SVG
from hailo_sdk_client import ClientRunner
import argparse
import json
import os



parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("-c", "--cfg", help="ONNX 2 HAR configuration file as .json file", default='.json')
args = parser.parse_args()
f = open(args.cfg)
data = json.load(f)
chosen_hw_arch = data['hailo_arch']
onnx_path = data['onnx_path']
print(chosen_hw_arch)
print(onnx_path)
f.close()

if not os.path.exists(data['output_path']):
    os.mkdir(data['output_path'])

# Read input size of model
size_input = data['size_input']
size_list = size_input.split(",")
size_input = [int(x) for x in size_list]
print("input size=", size_input)

# Run Hailo Translator
runner = ClientRunner(hw_arch=chosen_hw_arch)
hn, npz = runner.translate_onnx_model(
    onnx_path,
    data["model_name"],
    start_node_names=data["name_start"],
    end_node_names=data["name_end"],
    net_input_shapes={data["name_start"]: size_input},
)

runner.save_har(data["output_path"] + data["model_name"] + ".har")
