from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


model = tf.keras.models.load_model("models")
img_path = "processed_Test/E_test.jpg"  # Example image path
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

# Predict
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions)

# Map index back to class name
class_indices = {
    'A': 0,
    'B': 1,
    'Bye': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7,
    'H': 8,
    'Hello': 9,
    'I': 10,
    'J': 11,
    'K': 12,
    'L': 13,
    'M': 14,
    'N': 15,
    'No': 16,
    'O': 17,
    'P': 18,
    'Perfect': 19,
    'Q': 20,
    'R': 21,
    'S': 22,
    'T': 23,
    'Thank You': 24,
    'U': 25,
    'V': 26,
    'W': 27,
    'X': 28,
    'Y': 29,
    'Yes': 30,
    'Z': 31,
    'del': 32,
    'nothing': 33,
    'space': 34
}
class_names = list(class_indices.keys())
print("Predicted sign:", class_names[predicted_class_index])

test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)

test_generator = test_datagen.flow_from_directory(
    "processed_Test",
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    shuffle=False
)

loss, acc = model.evaluate(test_generator)
# print(f"Validation Accuracy: {acc * 100:.2f}%")

for i in range(5):
    image_batch, label_batch = next(iter("processed_Test"))
    prediction = model.predict(image_batch)
    predicted_class = np.argmax(prediction[0])

    plt.imshow(image_batch[0].astype("uint8"))
    plt.title(f"Predicted: {class_names[predicted_class]} | Actual: {class_names[int(label_batch[0])]}")
    plt.axis('off')
    plt.show()