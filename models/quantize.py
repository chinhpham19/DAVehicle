from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'license_plate_det.onnx'
model_int8 = 'license_plate_det_quantized.onnx'

# Quantize
quantize_dynamic(model_fp32, model_int8, weight_type=QuantType.QUInt8)