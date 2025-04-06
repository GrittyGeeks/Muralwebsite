from flask import Flask, request, render_template, jsonify
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

def validate_image(file_path):
    """Verify image can be processed"""
    try:
        img = Image.open(file_path)
        img.verify()
        img.close()
        return True
    except Exception as e:
        print(f"Invalid image: {str(e)}")
        return False

def classify_image(img_path):
    """Enhanced image classification with validation"""
    try:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")
        
        print(f"Classifying image: {img_path}")
        img = keras_image.load_img(img_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        prediction = model.predict(img_array)[0][0]
        print(f"Raw prediction value: {prediction}")
        return prediction > 0.5
    except Exception as e:
        print(f"Classification error: {traceback.format_exc()}")
        raise

def clahe_restore(img_path):
    """Enhanced CLAHE restoration"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("CV2 failed to read image")
        
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    except Exception as e:
        print(f"CLAHE restoration failed: {traceback.format_exc()}")
        raise

def inpaint_restore(img_path):
    """Enhanced inpainting"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("CV2 failed to read image")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        return cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
    except Exception as e:
        print(f"Inpainting failed: {traceback.format_exc()}")
        raise

@app.route('/classify', methods=['POST'])
def classify_endpoint():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        
        filename = secure_filename(file.filename)
        temp_path = os.path.join("/tmp", filename)
        
        try:
            file.save(temp_path)
            if not validate_image(temp_path):
                return jsonify({'error': 'Invalid image file'}), 400
            
            is_faded = classify_image(temp_path)
            classification = "Faded" if is_faded else "Cracked"
            
            file.seek(0)
            image_url = upload_to_supabase(file, "uploads", filename)
            
            return jsonify({
                'status': 'success',
                'classification': classification,
                'image_url': image_url
            })
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        print(f"Endpoint error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/restore', methods=['POST'])
def restore_endpoint():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        
        filename = secure_filename(file.filename)
        temp_path = os.path.join("/tmp", filename)
        
        try:
            file.save(temp_path)
            if not validate_image(temp_path):
                return jsonify({'error': 'Invalid image file'}), 400
            
            is_faded = classify_image(temp_path)
            restored_img = clahe_restore(temp_path) if is_faded else inpaint_restore(temp_path)
            method = "CLAHE" if is_faded else "Inpainting"
            
            _, buffer = cv2.imencode('.jpg', restored_img)
            restored_bytes = io.BytesIO(buffer).getvalue()
            
            restored_filename = f"restored_{filename}"
            restored_url = upload_to_supabase(restored_bytes, "restored", restored_filename)
            
            return jsonify({
                'status': 'success',
                'restored_image': restored_url,
                'method': method
            })
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        print(f"Restoration error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/map')
def map():
    return render_template('map.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
