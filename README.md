
**AI Future Directions Assignment – Pioneering Tomorrow's AI Innovations**

**Overview**
This project explores future applications of Artificial Intelligence (AI) through two main tasks:

* Task 1: Edge AI for Smart Waste Classification
* Task 2: AI and IoT Integration for Smart Agriculture

The goal is to apply AI development workflows, optimize real-world deployment, and explore the ethical and social impact of responsible AI.

---

**Objectives**

* Train an AI model to classify real-world images
* Convert and deploy the model for edge devices like Raspberry Pi
* Design an AI-powered IoT system for farming
* Reflect on ethical and technical challenges

---

**Task 1 – Edge AI Waste Classification**
This task focuses on training a small convolutional neural network (CNN) that can classify recyclable items: Plastic, Paper, and Glass. The model is trained using TensorFlow, then converted to TensorFlow Lite for low-power deployment.

Model code:

```python
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])
```

Inference example:

```python
img = Image.open('test.jpg').resize((150, 150))
img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

prediction = model.predict(img_array)
print("Predicted class:", np.argmax(prediction))
```

Plot accuracy:

```python
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')
```

Benefits of using Edge AI:

* Works without internet
* Faster predictions (low latency)
* Saves power and memory
* Protects user privacy

---

**Task 2 – Smart Agriculture with AI + IoT**
This task designs a smart farming system using sensors and a simple machine learning model to predict crop yield based on environment data.

Prediction model:

```python
from sklearn.linear_model import LinearRegression

X = [[30,25,200],[60,22,180],[45,20,100],[70,18,220]]
y = [1.2, 2.0, 1.8, 2.5]

model = LinearRegression()
model.fit(X, y)

new_data = [[50, 21, 190]]
print("Predicted crop yield:", model.predict(new_data))
```

Sensors used:

* Soil moisture
* Temperature (DHT11)
* Light (LDR)
* Humidity
* pH
* Rain sensor

These sensors send data to a microcontroller like ESP32, which can pass the data to the AI model for prediction.

---

**Tools Used**

* Python 3.10
* TensorFlow 2.12.0
* Scikit-learn
* NumPy
* Matplotlib
* VS Code and Google Colab

To install required libraries:

```bash
pip install tensorflow==2.12.0 scikit-learn numpy matplotlib
```

---

**Ethical Focus**
This work supports two global goals:

* SDG 12: Responsible Consumption
* SDG 2: Zero Hunger

Both projects are designed to be useful in areas with limited internet and hardware access.

---

**Reflections and Challenges**

* It was challenging to reduce model size without losing too much accuracy
* Creating the dataset helped understand the real difficulty of collecting usable data
* The agriculture task highlighted the power of AI when combined with simple sensors

---
