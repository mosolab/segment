import os
from flask import Flask, send_file, request, jsonify
from run import get_mask
import cv2

IMG_PATH = 'assets/test_imgs/'
OUT_PATH = 'out/'
weights_path = './weights/sbd_vit_xtiny.pth'
device = 'cpu'

app = Flask(__name__)
app.secret_key = "mask"

@app.route("/", methods=["GET"])
def home():
    return "Test"

@app.route("/mask", methods=["POST"])
def mask_img():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Retrieve x and y coordinates from the request data
    x = request.form.get('x')
    y = request.form.get('y')
    if x is None or y is None:
        return jsonify({'error': 'Coordinates x or y are missing'}), 400

    try:
        # Convert coordinates to integer values
        x = int(x)
        y = int(y)
    except ValueError:
        return jsonify({'error': 'Invalid coordinates, must be integers'}), 400

    filepath = os.path.join(IMG_PATH, file.filename)
    file.save(filepath)  # Save the uploaded file

    # Call the image processing function
    merged_img = get_mask(filepath, weights_path, device, x, y)

    if merged_img is None:
        return jsonify({'error': 'Failed to process image'}), 500

    resname = os.path.join(OUT_PATH, file.filename)
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
    cv2.imwrite(resname, merged_img)  # Save the processed image

    return jsonify({'mask_path': resname}) #send_file(resname, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run('0.0.0.0')
