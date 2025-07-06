import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

# Load images from dataset/
train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_gen.flow_from_directory(
    'dataset/',
    target_size=(150, 150),
    batch_size=4,
    class_mode='categorical',
    subset='training'
)

val_data = train_gen.flow_from_directory(
    'dataset/',
    target_size=(150, 150),
    batch_size=4,
    class_mode='categorical',
    subset='validation'
)

# Build MobileNetV2 model
base_model = MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
for layer in base_model.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(64, activation='relu')(x)
output = Dense(train_data.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, validation_data=val_data, epochs=5)

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("training_accuracy.png")
plt.close()

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Model converted and saved as model.tflite")

# Load and test one image (adjust path as needed)
img_path = 'dataset/plastic/plastic1.jpg'  # <- change this if needed
test_img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(test_img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output)

print("Prediction:", output)
print("Predicted class index:", predicted_class)
