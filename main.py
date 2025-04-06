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
import traceback

app = Flask(__name__, template_folder='templates')

# Supabase Configuration
SUPABASE_URL = "https://dnlzklnwenoraeldzrli.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRubHprbG53ZW5vcmFlbGR6cmxpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDM5MTczMjAsImV4cCI6MjA1OTQ5MzMyMH0.ElBQld5a6J-2fjJ5RjTJtCZgDi48MunO1GoEKz_n-eU"
BUCKET_NAME = "mural-restoration"

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Model Configuration
MODEL_PATH = "transfer_learning_vgg16_mural_model.h5"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_or_build_model():
    """Load model or build architecture with enhanced error handling"""
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Model loaded successfully. Input shape: {model.input_shape}")
            return model
        
        print("Building new model architecture...")
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
        
        return model
    except Exception as e:
        print(f"Model initialization failed: {traceback.format_exc()}")
        raise RuntimeError(f"Model initialization failed: {str(e)}")

# Initialize model
model = load_or_build_model()

def upload_to_supabase(file, folder, filename):
    """Enhanced file upload with validation"""
    try:
        file_path = f"{folder}/{filename}"
        file_bytes = file.read() if hasattr(file, 'read') else file
        
        print(f"Attempting to upload {len(file_bytes)} bytes to {file_path}")
        res = supabase.storage.from_(BUCKET_NAME).upload(file_path, file_bytes)
        
        if res.status_code != 200:
            raise Exception(f"Upload failed with status {res.status_code}: {res.error}")
        
        url = supabase.storage.from_(BUCKET_NAME).get_public_url(file_path)
        print(f"Upload successful. Public URL: {url}")
        return url
    except Exception as e:
        print(f"Upload error: {traceback.format_exc()}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/map')
def map_page():  # Renamed to avoid conflict with Python's map function
    return render_template('map.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'GET':
        return render_template('classify.html')
    
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    try:
        filename = secure_filename(file.filename)
        temp_path = os.path.join("/tmp", filename)
        file.save(temp_path)
        
        # Verify image
        try:
            img = Image.open(temp_path)
            img.verify()
            img.close()
        except Exception as e:
            return jsonify({'error': f'Invalid image: {str(e)}'}), 400
        
        # Classify
        is_faded = classify_image(temp_path)
        classification = "Faded" if is_faded else "Cracked"
        
        # Upload
        file.seek(0)
        image_url = upload_to_supabase(file, "uploads", filename)
        
        return jsonify({
            'status': 'success',
            'classification': classification,
            'image_url': image_url
        })
    except Exception as e:
        print(f"Error in classify: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/restore', methods=['GET', 'POST'])
def restore():
    if request.method == 'GET':
        return render_template('restore.html')  # Changed from app.html to restore.html
    
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    try:
        filename = secure_filename(file.filename)
        temp_path = os.path.join("/tmp", filename)
        file.save(temp_path)
        
        # Verify image
        try:
            img = Image.open(temp_path)
            img.verify()
            img.close()
        except Exception as e:
            return jsonify({'error': f'Invalid image: {str(e)}'}), 400
        
        # Process
        is_faded = classify_image(temp_path)
        restored_img = clahe_restore(temp_path) if is_faded else inpaint_restore(temp_path)
        method = "CLAHE" if is_faded else "Inpainting"
        
        # Save and upload
        _, buffer = cv2.imencode('.jpg', restored_img)
        restored_bytes = io.BytesIO(buffer).getvalue()
        restored_url = upload_to_supabase(restored_bytes, "restored", f"restored_{filename}")
        
        return jsonify({
            'status': 'success',
            'restored_image': restored_url,
            'method': method
        })
    except Exception as e:
        print(f"Error in restore: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def classify_image(img_path):
    """Image classification with error handling"""
    try:
        img = keras_image.load_img(img_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return model.predict(img_array)[0][0] > 0.5
    except Exception as e:
        print(f"Classification error: {traceback.format_exc()}")
        raise

def clahe_restore(img_path):
    """CLAHE restoration with error handling"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Failed to read image")
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    except Exception as e:
        print(f"CLAHE error: {traceback.format_exc()}")
        raise

def inpaint_restore(img_path):
    """Inpainting restoration with error handling"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Failed to read image")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        return cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
    except Exception as e:
        print(f"Inpainting error: {traceback.format_exc()}")
        raise

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
