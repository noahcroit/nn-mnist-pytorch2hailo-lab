from multiprocessing import Process
import argparse
import numpy as np
from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoSchedulingAlgorithm,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    InputVStreams,
    OutputVStreamParams,
    OutputVStreams,
    VDevice,
)
from hailo_sdk_client import ClientRunner, InferenceContext




def run_inference(input_data):
    global network_group
    global input_vstreams_params
    global output_vstreams_params
    infer_result=None

    # Running inference
    input_data = {input_vstream_info.name: dataset}
    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        with network_group.activate(network_group_params):
            infer_results = infer_pipeline.infer(input_data)
            # The result output tensor is infer_results[output_vstream_info.name]
            print(f"Stream output shape is {infer_results[output_vstream_info.name].shape}")

    return infer_result



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument("-e", "--hef", help="Hailo Executable File (.hef)", default='model.hef')
    parser.add_argument("-n", "--name", help="Model name", default='model')
    parser.add_argument("-m", "--mode", help="test with local images (test) or live-stream (capture)", default='test')
    args = parser.parse_args()

    # Setting VDevice params to disable the HailoRT service feature
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.NONE

    # The target can be used as a context manager (”with” statement) to ensure it's released on time.
    # Here it's avoided for the sake of simplicity
    target = VDevice(params=params)
    # Loading compiled HEFs to device:
    model_name = args.name
    hef_path = args.hef
    hef = HEF(hef_path)

    # Get the ”network groups” (connectivity groups, aka. ”different networks”) ,information from the .hef
    configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, configure_params)
    network_group = network_groups[0]
    network_group_params = network_group.create_params()

    # Create input and output virtual streams params
    # Quantized argument signifies whether or not the incoming data is already quantized.
    # Data is quantized by HailoRT if and only if quantized == False .
    input_vstreams_params = InputVStreamParams.make(network_group, quantized=False, format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make(network_group, quantized=True, format_type=FormatType.UINT8)

    if args.mode == 'test':
        # Define dataset params
        input_vstream_info = hef.get_input_vstream_infos()[0]
        output_vstream_info = hef.get_output_vstream_infos()[0]
        print(input_vstream_info.shape)

        # run inference to image data
        #ret = run_inference(image)


    elif args.mode == 'capture':
        pass
