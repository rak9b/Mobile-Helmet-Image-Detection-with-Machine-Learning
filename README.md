


# Mobile & Helmet Image Detection with Machine Learning

This project leverages machine learning techniques to detect images of mobile phones and helmets. It uses a deep learning model trained in Google Colab to classify whether an image contains a mobile phone, a helmet, or neither. The model can be applied to a variety of real-world applications, such as safety monitoring or mobile usage detection.

---

## 📋 Features

- **Mobile Detection**: Classify whether an image contains a mobile phone.
- **Helmet Detection**: Classify whether an image contains a helmet.
- **Real-Time Usage**: You can upload images to the trained model and receive predictions on whether they contain a mobile or helmet.
- **Field Application**: Can be used for safety monitoring, vehicle inspections, or any scenario requiring the detection of these objects.

---

## 🚀 Getting Started

### Prerequisites

Before you begin, make sure you have the following:

- **Python** (>=3.6)
- **Google Colab** account (used for training and inference)

### Installation

Since this project is primarily based on Google Colab, you do not need to set up your local machine environment. However, you should have access to Google Colab.

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/mobile-helmet-detection.git
    cd mobile-helmet-detection
    ```

2. **Dependencies**:
    The required libraries are installed directly in the Colab notebook. These libraries include TensorFlow, Keras, OpenCV, NumPy, etc.

    You can install the dependencies within Colab using the following commands:
    ```python
    !pip install tensorflow keras opencv-python numpy matplotlib
    ```

---

## 🧠 Model Training

The model for detecting mobile and helmet images was trained on a labeled dataset. To train the model using Google Colab:

1. **Open the Colab Notebook**: The main model training and inference are defined in the notebook `model_training_inference.ipynb`. Open this file in Google Colab.

2. **Dataset Preparation**: Use a labeled dataset of images containing mobiles and helmets. You can upload your dataset to Colab or use a public dataset.

3. **Model Training**:
    - Load the dataset into Colab.
    - Preprocess the images (resize, normalize, etc.).
    - Define and train the neural network model (e.g., using a Convolutional Neural Network (CNN)).
    - Evaluate the model’s performance.

4. **Saving the Model**: After training, the model is saved in `.h5` format:
    ```python
    model.save('mobile_helmet_model.h5')
    ```

5. **Testing the Model**: Use the trained model to make predictions on new images.

---

## 🔍 Using the Model for Inference

Once the model is trained, you can use it to classify new images of mobile phones and helmets. You can upload an image and get predictions whether it contains a mobile or a helmet.

### Steps to Use the Model:

1. **Upload Your Image**:
    Use the `upload_image` function in Colab to upload a new image for inference:
    ```python
    from google.colab import files
    uploaded = files.upload()  # Upload an image file
    ```

2. **Load the Model**:
    Load the saved model from the `.h5` file:
    ```python
    from tensorflow.keras.models import load_model
    model = load_model('mobile_helmet_model.h5')
    ```

3. **Preprocess the Image**:
    Preprocess the uploaded image (resize, normalize, etc.):
    ```python
    import cv2
    import numpy as np

    # Load and preprocess the image
    image = cv2.imread('your_image.jpg')  # Replace with your image filename
    image = cv2.resize(image, (224, 224))  # Resize to the input size of the model
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize the image
    ```

4. **Make Predictions**:
    Use the model to predict the contents of the image:
    ```python
    prediction = model.predict(image)

    if prediction[0][0] > 0.5:
        print("The image contains a Mobile Phone.")
    else:
        print("The image does not contain a Mobile Phone.")

    if prediction[0][1] > 0.5:
        print("The image contains a Helmet.")
    else:
        print("The image does not contain a Helmet.")
    ```

5. **Display the Image and Result**:
    Display the image with the prediction result:
    ```python
    import matplotlib.pyplot as plt

    plt.imshow(image[0])
    plt.title(f"Prediction: {'Mobile Phone' if prediction[0][0] > 0.5 else 'No Mobile Phone'}, {'Helmet' if prediction[0][1] > 0.5 else 'No Helmet'}")
    plt.show()
    ```

---

## 📱 Using the Model in the Field

To use this model in real-world applications (e.g., safety monitoring or vehicle inspections), you can integrate the trained model into various platforms:

### 1. **Mobile Application**:
    - Convert the model into a TensorFlow Lite format (`.tflite`).
    - Use it in Android/iOS apps for real-time image classification using the phone's camera.

    **Convert the Model**:
    ```python
    import tensorflow as tf

    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model
    with open('mobile_helmet_model.tflite', 'wb') as f:
        f.write(tflite_model)
    ```

    Use the `.tflite` model in a mobile application to detect helmets and mobile phones from the camera in real-time.

### 2. **Web Application**:
    - Integrate the model with a web service using Flask or FastAPI for serving predictions.
    - Allow users to upload images, which are processed by the model and the prediction is returned.

### 3. **Edge Devices**:
    - Deploy the model to edge devices (such as cameras or IoT systems) to detect mobiles and helmets in real time.

---

## 🧪 Testing

The model can be tested using images in a local environment. Once the model is loaded, you can provide test images for inference, as demonstrated above.

### Run Unit Tests
If you'd like to run automated tests to verify the model's performance or inference logic, you can use the `unittest` module in Python.

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the model or extend the application, feel free to fork this repository, create a new branch, and submit a pull request.

### Steps for Contributing:
1. Fork the repository
2. Create a new branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request

---

## 📝 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

