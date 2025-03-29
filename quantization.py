import model_compression_toolkit as mct
from model_compression_toolkit.core import QuantizationErrorMethod
from dataSetGenerator import representative_dataset_gen
from typing import Generator
from tensorflow.keras.models import load_model


train_dir = 'processed_Train'
MODEL_PATH = "models/"  
float_model = load_model(MODEL_PATH)
n_iter=10

# Create representative dataset generator
def get_representative_dataset() -> Generator:
    """A function that loads the dataset and returns a representative dataset generator.

    Returns:
        Generator: A generator yielding batches of preprocessed images.
    """
    dataset = train_dir

    def representative_dataset() -> Generator:
        """A generator function that yields batch of preprocessed images.

        Yields:
            A batch of preprocessed images.
        """
        for _ in range(n_iter):
            yield dataset.take(1).get_single_element()[0].numpy()

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

