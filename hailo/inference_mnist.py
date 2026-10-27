import argparse
import numpy as np
import cv2
import queue
import threading
from time import process_time_ns



def image_preprocess(frame_in):
    # Thresholding to remove background
    tmp = cv2.cvtColor(frame_in, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(tmp, 100, 255, cv2.THRESH_BINARY)
    frame_in = cv2.bitwise_or(frame_in, frame_in, mask=mask)

    # Increase the contrast of input image
    ycrcb_image = cv2.cvtColor(frame_in, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)
    y_channel_stretched = cv2.normalize(y_channel, None, 0, 255, cv2.NORM_MINMAX)
    contrast_stretched_ycrcb = cv2.merge([y_channel_stretched, cr_channel, cb_channel])
    frame_in = cv2.cvtColor(contrast_stretched_ycrcb, cv2.COLOR_YCrCb2BGR)

    # Invert and Normalize the pixel values
    frame_in = cv2.bitwise_not(frame_in)

    # Crop only ROI, Resize to 28x28 as the input size of ANN model
    h, w, _ = frame_in.shape
    l = np.min([h, w])
    start_x = w // 2 - l // 2
    start_y = h // 2 - l // 2
    end_x = start_x + l
    end_y = start_y + l
    frame_in = frame_in[start_y:end_y, start_x:end_x]
    frame_out = cv2.resize(frame_in, (28, 28), interpolation=cv2.INTER_AREA)
    frame_out = frame_out / 256
    return frame_out

def task_queueframe():
    global q
    global stream
    while stream.isOpened():
        # read frame
        ret, frame = stream.read()
        if not ret:
            break

        # put frame to queue
        q.put(frame)
    stream.release()

def run_inference(x, use_hailo=True):
    global network_group
    global input_vstreams_params
    global output_vstreams_params

    # Running inference
    if use_hailo:
        x = np.expand_dims(x, axis=0)
        input_data = {input_vstream_info.name: x}
        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            with network_group.activate(network_group_params):
                infer_results = infer_pipeline.infer(input_data)
                # The result output tensor is infer_results[output_vstream_info.name]
                # print(f"Stream output shape is {infer_results[output_vstream_info.name].shape}")
        # make decision from output with max()
        # drop the softmax function since the predict outputs are 8-bit INT, easy to find max with argmax()
        predict = np.argmax(infer_results[output_vstream_info.name])
    else:
        x = np.expand_dims(x, axis=0)
        predict = np.argmax(session.run(None, {'input': x}))
    return predict 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument("-e", "--model", help="Onnx model (.onnx) or Hailo Executable File (.hef)", default='model.hef')
    parser.add_argument("-n", "--name", help="Model name", default='model')
    parser.add_argument("-s", "--source", help="test with local images (local) or live-stream (live)", default='local')
    parser.add_argument("-m", "--mode", help="onnx or hailo", default='hailo')
    args = parser.parse_args()

    if args.mode == 'hailo':
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
        # Setting VDevice params to disable the HailoRT service feature
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.NONE

        # The target can be used as a context manager (”with” statement) to ensure it's released on time.
        # Here it's avoided for the sake of simplicity
        target = VDevice(params=params)
        # Loading compiled HEFs to device:
        model_name = args.name
        hef_path = args.model
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

        # Define dataset params
        input_vstream_info = hef.get_input_vstream_infos()[0]
        output_vstream_info = hef.get_output_vstream_infos()[0]
        print(input_vstream_info.shape)
        print(output_vstream_info.shape)

    elif args.mode == 'onnx':
        import onnxruntime
        # Load the ONNX model
        session = onnxruntime.InferenceSession(args.model)
        
    if args.source == 'local':
        # inference with local images using opencv
        for number in range(10):
            # read image
            img_path = '../img/' + str(number) + '.jpg'
            img = cv2.imread(img_path)
            cv2.imshow("input number", img)
            cv2.waitKey(1500)
            
            # preprocess image
            img = image_preprocess(img)
            img = img[:,:,0]
            img = img.astype(np.float32)
            
            # apply Flattening()
            x = img.flatten()
            
            # run hailo inference to flattened image data
            t_start = process_time_ns()
            if args.mode == 'onnx':
                predict = run_inference(x, use_hailo=False)
            else:
                predict = run_inference(x)
            print("run inference with the picture from ", img_path, ", Predicted Num = ", predict)

    elif args.source == 'live':
        # inference with webcamera using opencv
        stream = cv2.VideoCapture(0)
        stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280);
        stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720);
        q = queue.Queue()
        t1 = threading.Thread(target=task_queueframe)
        t1.start()
        fps_previous=0
        t_previous = process_time_ns()
        while stream.isOpened():
            # drop old frames and read the recent frame
            frame_cnt = q.qsize()
            if frame_cnt != 0:
                for i in range(frame_cnt):
                    frame = q.get()
                
                # preprocess image
                frame_out = image_preprocess(frame)
                frame_out = frame_out[:,:,0]
                frame_out = frame_out.astype(np.float32)
                
                # apply Flattening()
                x = frame_out.flatten()

                # run hailo inference to flattened image data
                t_start = process_time_ns()
                if args.mode == 'onnx':
                    predict = run_inference(x, use_hailo=False)
                else:
                    predict = run_inference(x)
                
                # FPS measurement
                t_current = process_time_ns()
                fps = 1/(t_current - t_previous)*1000000000
                t_previous = t_current
                fps = 0.75*fps_previous + 0.25*fps
                fps_previous = fps
                print("FPS=", round(fps, 1))

                # output / display
                font = cv2.FONT_HERSHEY_SIMPLEX
                h, w, _ = frame.shape
                txcolor= (0, 255, 255)
                cv2.putText(frame, "FPS="+str(round(fps, 1)), (w - int(w*0.3), int(h*0.2)), font, 2, txcolor, 4, cv2.LINE_AA)
                cv2.putText(frame, str(predict), (0, h - 40), font, 10, txcolor, 4, cv2.LINE_AA)
                cv2.imshow("input", frame)
                cv2.imshow("output", frame_out)

                #check keyboard for exit program
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'):
                    break
        stream.release()
