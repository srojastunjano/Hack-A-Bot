import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

# Constants
MODEL = "models"
BATCH_SIZE = 32
IMAGE_SHAPE = (224, 224)
train_dir = 'processed_Train'

# Set up data generator with 80/20 split
train_datagen = ImageDataGenerator(
    validation_split=0.2,
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

# Load training data (80%)
train_ds = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SHAPE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training',
    shuffle=True
)

# Load validation data (20%)
validation_ds = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SHAPE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation',
    shuffle=True
)

# Get class names and number of classes
class_names = list(train_ds.class_indices.keys())
num_classes = len(class_names)


# MobileNet_V2

# Loads the MobileNetV2 model.
# weights='imagenet': Uses pre-trained knowledge from ImageNet (a large dataset of images).
# include_top=False: Removes the default final classification layer, so you can add your own custom output.
# input_shape=IMAGE_SHAPE+(3,): Sets input image size (e.g., 224×224×3). The (3,) means the image has 3 color channels (RGB).

base_model =  MobileNetV2(weights='imagenet', include_top=False, input_shape=IMAGE_SHAPE+(3,))

# A neuron: fundamental building block, It takes inputs, processes them, and produces an output used for predictions.

# gets the last layer (feature vector without classification)
x = base_model.output

# compresses feature vector by averaging
x = GlobalAveragePooling2D()(x)

# Adds a final classification layer:
# Dense(num_classes): One neuron per class (e.g., 3 if you're classifying A, B, C).
# activation='softmax': Turns the outputs into probabilities (e.g., "90% sure this is C").
predictions = Dense(num_classes, activation=tf.nn.softmax)(x)

# combines the MobileNet feature vector with the pooling or classifying layers to create a model ready to train.
float_model = Model(inputs=base_model.input, outputs=predictions)

# Freeze layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# float_model.summary()

# Display dataset information
print("class_names:", class_names)
print("num_classes:", num_classes)
print("Training batches:", len(train_ds))
print("Validation batches:", len(validation_ds))


# # training data

# EPOCHS = 5

# float_model.compile(
#     optimizer=tf.keras.optimizers.legacy.Adam(),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     metrics=['accuracy']
# )

# callback = tf.keras.callbacks.EarlyStopping(
#     monitor="val_accuracy",
#     baseline=0.8,
#     min_delta=0.01,
#     mode='max',
#     patience=5,
#     verbose=1,
#     restore_best_weights=True,
#     start_from_epoch=5,
# )

# history = float_model.fit(
#     train_ds,
#     validation_data=validation_ds,
#     epochs=EPOCHS,
#     callbacks=[callback]
# )

# float_model.save(MODEL)