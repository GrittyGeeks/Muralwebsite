from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configuration
UPLOAD_FOLDER = "/tmp/uploads"
RESTORE_FOLDER = "/tmp/restored"
MODEL_PATH = "transfer_learning_vgg16_mural_model.h5"


# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESTORE_FOLDER, exist_ok=True)


from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/restored/<filename>')
def restored_file(filename):
    return send_from_directory(RESTORE_FOLDER, filename)


# =============================================
# MODEL LOADING
# =============================================

def load_or_build_model():
    """Load model or build architecture"""
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully with full architecture")
            return model
        except Exception as e:
            print(f"Couldn't load full model: {str(e)}")
    
    # Fallback: Build architecture and try loading weights
    print("Building model architecture to match weights")
    try:
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=x)

        if os.path.exists(MODEL_PATH):
            model.load_weights(MODEL_PATH)
            print("Weights loaded successfully")
        else:
            print("No weights file found at MODEL_PATH")

        return model
    except Exception as e:
        print(f"Critical error building model: {str(e)}")
        raise RuntimeError("Could not initialize model") from e

# Initialize model when app starts
model = load_or_build_model()

# =============================================
# ROUTES
# =============================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/map')
def map():
    return render_template('map.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'GET':
        return render_template('classify.html')
    
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        filename = secure_filename(file.filename)
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(img_path)
        
        is_faded = classify_image(img_path)
        classification = "Faded" if is_faded else "Cracked"
        
        return jsonify({
            'status': 'success',
            'classification': classification,
            'image_url': f"/static/uploads/{filename}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/app', methods=['GET', 'POST'])
def restore():
    if request.method == 'GET':
        return render_template('app.html')
    
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        filename = secure_filename(file.filename)
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(img_path)
        
        img_class = classify_image(img_path)
        restored_img = clahe_restore(img_path) if img_class else inpaint_restore(img_path)
        method = "CLAHE" if img_class else "Navier Stokes"
        
        restored_filename = f"restored_{filename}"
        restored_path = os.path.join(RESTORE_FOLDER, restored_filename)
        cv2.imwrite(restored_path, restored_img)
        
        return jsonify({
            'status': 'success',
            'restored_image': f"/static/restored/{restored_filename}",
            'method': method
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================
# IMAGE PROCESSING FUNCTIONS
# =============================================

def classify_image(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return model.predict(img_array)[0][0] > 0.5

def clahe_restore(img_path):
    img = cv2.imread(img_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def inpaint_restore(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
