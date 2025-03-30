import model_compression_toolkit as mct
from model_compression_toolkit.core import QuantizationErrorMethod
from dataSetGenerator import representative_dataset_gen
from typing import Generator
from tensorflow.keras.models import load_model
import tensorflow as tf


train_dir = 'processed_Train'
MODEL_PATH = "model_1/"  
float_model = load_model(MODEL_PATH)
n_iter=10

# Create representative dataset generator
def get_representative_dataset():
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory='processed_Train',
        labels=None,
        image_size=(224, 224),  # Use your model's input size
        batch_size=1
    ).map(lambda x: x / 255.0)  # Normalize if needed

    def representative_dataset():
        for input_value in dataset.take(10):  # or whatever n_iter you want
            yield [input_value.numpy()]  # Must return a list or tuple of numpy arrays

    return representative_dataset

# Create a representative dataset generator
representative_dataset_gen = get_representative_dataset()

# Specify the IMX500-v1 target platform capability (TPC)
tpc = mct.get_target_platform_capabilities("tensorflow", 'imx500', target_platform_version='v1')

# Set the following quantization configurations:
# Choose the desired QuantizationErrorMethod for the quantization parameters search.
# Enable weights bias correction induced by quantization.
# Enable shift negative corrections for improving 'signed' non-linear functions quantization (such as swish, prelu, etc.)
# Set the threshold to filter outliers with z_score of 16.
q_config = mct.core.QuantizationConfig(activation_error_method=QuantizationErrorMethod.MSE,
                                       weights_error_method=QuantizationErrorMethod.MSE,
                                       weights_bias_correction=True,
                                       shift_negative_activation_correction=True,
                                       z_threshold=16)

ptq_config = mct.core.CoreConfig(quantization_config=q_config)

quantized_model, quantization_info = mct.ptq.keras_post_training_quantization(
    in_model=float_model,
    representative_data_gen=representative_dataset_gen,
    core_config=ptq_config,
    target_platform_capabilities=tpc)

# Export the quantized model
mct.exporter.keras_export_model(model=quantized_model, save_model_path=MODEL_PATH)

converter = tf.lite.TFLiteConverter.from_saved_model("models/")
tflite_model = converter.convert()

with open("model_quantized.tflite", "wb") as f:
    f.write(tflite_model)

