import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K  # Import Keras backend for custom metric

# Define a Flask app
app = Flask(__name__)

# Load the model without compiling first
model = tf.keras.models.load_model('cnn_model.keras', compile=False)

# Custom F1 score metric
def f1_score(y_true, y_pred):
    # Count positive samples
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))  # predicted positives
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))  # true positives in the ground truth

    # Avoid division by zero by checking if c3 is zero
    c3_zero = K.equal(c3, 0)  # Check if c3 is zero

    # If there are no true samples, return zero F1 score
    precision = K.switch(c3_zero, K.zeros_like(c1), c1 / c2)
    recall = K.switch(c3_zero, K.zeros_like(c1), c1 / c3)

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())  # K.epsilon() to avoid division by zero
    return f1

# Compile the model with the custom metric
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1_score])

print('Model loaded and compiled. Check http://127.0.0.1:5000/')

# List of possible disease classes
disease_classes = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight", 
    "Corn_(maize)___healthy", "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy", 
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight", 
    "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", 
    "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", 
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", 
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

def model_predict(img_path, model):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image to match model input size
    x = image.img_to_array(img) / 255.0  # Convert to array and normalize
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    preds = model.predict(x)  # Make prediction using the model
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads directory
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction for the uploaded image
        preds = model_predict(file_path, model)

        # Get the predicted class index
        predicted_class = np.argmax(preds[0])  # Get the index of the highest predicted score

        print('Predicted class:', preds)

        return jsonify({
            'prediction': disease_classes[predicted_class],  # Use the disease class name as the prediction
            'accuracy': str(preds[0][predicted_class])  # Return the score of the predicted class
        })

    return None

if __name__ == '__main__':
    # Serve the app with gevent for better performance
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
    app.run()
