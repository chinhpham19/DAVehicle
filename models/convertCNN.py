import onnx
import os
import tf2onnx
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import LeakyReLU, Softmax

tf.keras.utils.get_custom_objects().update({'LeakyReLU': LeakyReLU, 'Softmax': Softmax})
model_path = 'model.h5'

# Load the saved model
try:
    model = load_model(model_path)
    # If the model is Sequential, convert it to Functional API
    if isinstance(model, tf.keras.Sequential):
        input_layer = layers.Input(shape=model.input_shape[1:], name='input_layer_1')
        output_layer = model(input_layer)
        model = Model(inputs=input_layer, outputs=output_layer)

    # Define the input signature for tf2onnx conversion
    input_signature = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name='input_layer_1')]

    # Convert the model to ONNX format
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
    output_model_path = "recognition.onnx"
    onnx.save(onnx_model, output_model_path)
except ValueError as e:
    print("Error:", e)