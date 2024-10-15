import onnx



onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
