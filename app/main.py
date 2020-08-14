from flask import Flask, request, jsonify
from app.torch_utils import tranform_image, get_prediction

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename=="":
            return jsonify({"error": "no file!"})
        if not allowed_file(file.filename):
            return jsonify({"error": "format not supported"})
        try:
            image_bytes = file.read()
            image_tensor = tranform_image(image_bytes)
            prediction = get_prediction(image_tensor)
            data = {
                "prediction": prediction.item(),
                "class_name": str(prediction.item())
            }
            return jsonify(data)
        except:
            return jsonify({"error": "error during prediction"})