from flask import Flask, request, render_template, jsonify, url_for
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from supabase import create_client, Client
import io
from PIL import Image

app = Flask(__name__, template_folder='templates')

# Supabase Configuration
SUPABASE_URL = "https://your-project-id.supabase.co"
SUPABASE_KEY = "your-anon-key"
BUCKET_NAME = "mural-restoration"

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Model Configuration
MODEL_PATH = "transfer_learning_vgg16_mural_model.h5"

# =============================================
# MODEL LOADING (Same as original)
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

# Initialize model on startup
model = load_or_build_model()

# =============================================
# SUPABASE STORAGE HELPER FUNCTIONS
# =============================================

def upload_to_supabase(file, folder, filename):
    """Upload file to Supabase Storage"""
    file_path = f"{folder}/{filename}"
    
    # For Flask file objects
    if hasattr(file, 'read'):
        file_bytes = file.read()
    else:
        file_bytes = file
    
    res = supabase.storage.from_(BUCKET_NAME).upload(file_path, file_bytes)
    if res.status_code != 200:
        raise Exception(f"Supabase upload failed: {res.error}")
    
    # Get public URL
    url_res = supabase.storage.from_(BUCKET_NAME).get_public_url(file_path)
    return url_res

def download_from_supabase(file_path):
    """Download file from Supabase Storage"""
    res = supabase.storage.from_(BUCKET_NAME).download(file_path)
    if isinstance(res, bytes):
        return res
    raise Exception(f"Supabase download failed: {res.error}")

# =============================================
# ROUTES (Modified for Supabase)
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
        
        # Temporarily save file for classification (could be optimized)
        temp_path = f"/tmp/{filename}"
        file.save(temp_path)
        
        is_faded = classify_image(temp_path)
        classification = "Faded" if is_faded else "Cracked"
        
        # Upload original image to Supabase
        file.seek(0)  # Reset file pointer
        image_url = upload_to_supabase(file, "uploads", filename)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return jsonify({
            'status': 'success',
            'classification': classification,
            'image_url': image_url
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
        
        # Temporarily save file for processing
        temp_path = f"/tmp/{filename}"
        file.save(temp_path)
        
        # Classify and restore
        img_class = classify_image(temp_path)
        restored_img = clahe_restore(temp_path) if img_class else inpaint_restore(temp_path)
        method = "CLAHE" if img_class else "Navier Stokes"
        
        # Save restored image to bytes
        _, buffer = cv2.imencode('.jpg', restored_img)
        restored_bytes = io.BytesIO(buffer).getvalue()
        
        # Upload restored image to Supabase
        restored_filename = f"restored_{filename}"
        restored_url = upload_to_supabase(
            restored_bytes, 
            "restored", 
            restored_filename
        )
        
        # Clean up temp file
        os.remove(temp_path)
        
        return jsonify({
            'status': 'success',
            'restored_image': restored_url,
            'method': method
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================
# IMAGE PROCESSING FUNCTIONS (Same as original)
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

if __name__ == '__main__':
    app.run(debug=True)
