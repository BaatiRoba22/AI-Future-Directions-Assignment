import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load and preprocess test image
img_path = "dataset/plastic/plastic1.jpg"  # or any other image
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])
predicted_index = np.argmax(output)

print("✅ Prediction probabilities:", output)
print("🔍 Predicted class index:", predicted_index)
