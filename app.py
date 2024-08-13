from flask import Flask, render_template, request, send_from_directory
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Define the folder where uploaded images will be stored
UPLOAD_FOLDER = 'C:\\Users\\Kiy\\Desktop\\Python Project\\project1\\cardiovascular risk prediction\\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Specify the absolute path to the model file
model_path = 'C:\\Users\\Kiy\\Desktop\\Python Project\\project1\\cardiovascular risk prediction\\model70.h5'

# Load your pre-trained Keras model
loaded_model = load_model(model_path)

# Function to process the uploaded image
def process_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(512, 512))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    return img

# Function to predict risk level
def predict_risk(image_path):
    # Process the uploaded image
    processed_image = process_image(image_path)

    # Make predictions
    result = loaded_model.predict(processed_image)
    hypertension_prob = result[0][0]

    # Define probability thresholds for each risk level
    high_risk_threshold = 0.7
    moderate_risk_threshold = 0.4
    low_risk_threshold = 0.2

    # Assign risk level based on probability thresholds
    if hypertension_prob > high_risk_threshold:
        risk_level = "High Risk"
    elif hypertension_prob > moderate_risk_threshold:
        risk_level = "Moderate Risk"
    elif hypertension_prob > low_risk_threshold:
        risk_level = "Low Risk"
    else:
        risk_level = "Very Low Risk"

    # Format the result with percentage
    percentage = f"{hypertension_prob * 100:.2f}%"
    return risk_level, percentage

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No file part"

    file = request.files['image']

    if file.filename == '':
        return "No selected file"

    if file:
        # Save the uploaded image to the designated folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Predict risk level
        risk_level, percentage = predict_risk(file_path)

        # Relative path to the uploaded file for display
        relative_file_path = os.path.join("static/uploads", file.filename)

        # Render the upload template with the result and image
        return render_template('upload.html', result=f"Image uploaded successfully. Predicted Risk Level: {risk_level} ({percentage})", image_path=relative_file_path)

# Serve static files from the uploads folder
@app.route('/static/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
