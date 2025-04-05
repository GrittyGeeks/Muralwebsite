from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static', template_folder='templates')

UPLOAD_FOLDER = "static/uploads"
RESTORE_FOLDER = "static/restored"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESTORE_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/map')
def map_page():
    return render_template('map.html')


@app.route('/restore', methods=['GET', 'POST'])
def restore():
    if request.method == 'GET':
        return render_template('restore.html')

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    img_type = request.form.get('type')

    filename = secure_filename(image.filename)
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(img_path)

    restored_img, method_name = restore_image(img_path, img_type)

    restored_filename = f"restored_{filename}"
    restored_path = os.path.join(RESTORE_FOLDER, restored_filename)
    cv2.imwrite(restored_path, restored_img)

    restored_url = f"/static/restored/{restored_filename}"

    return jsonify({
        'restored_image': restored_url,
        'method_name': method_name
    })

def restore_image(img_path, img_type):
    img = cv2.imread(img_path)

    if img_type == "faded":
        restored_img = clahe_only(img.copy())
        method_name = "CLAHE Restoration"

    elif img_type == "cracked":
        restored_img = crack_restoration(img.copy())
        method_name = "Crack Restoration (Navier-Stokes)"

    else:
        return img, "Unknown method"

    return restored_img, method_name


def clahe_only(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    merged = cv2.merge((l_clahe, a, b))
    restored_image = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return restored_image


def crack_restoration(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    mask = mask.astype(np.uint8)

    inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

    return inpainted

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
