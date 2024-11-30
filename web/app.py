from flask import Flask, render_template, request, jsonify
from PIL import Image
from PIL.ImageOps import invert
import os
import sys
import torch

sys.path.append(os.getcwd())

from src.models.cnn import shufflenet_model 
from src.data import SiameseDataset

model = shufflenet_model().to('cuda')
model.load_state_dict(torch.load('models/shufflenet_best.pth'))
model.eval()

app = Flask(__name__)

# Ensure upload directory exists
UPLOAD_FOLDER = 'web/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

transform = SiameseDataset.get_transform(232, 224, False)

def verify_signatures(original_path, test_path):
    """
    Compare two signatures using the loaded model and return a result.
    """
    original_img = invert(Image.open(original_path).convert('L'))
    test_img = invert(Image.open(test_path).convert('L'))
    original_img = transform(original_img)  
    test_img = transform(test_img)          

    original_img = original_img.unsqueeze(0).to('cuda')
    test_img = test_img.unsqueeze(0).to('cuda')

    output = model(original_img, test_img)  
    distance = output.item()
    if distance > 0.242:
        return False, distance
    return True, distance

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    # Check if both files are uploaded
    if 'original_signature' not in request.files or 'verification_signature' not in request.files:
        return jsonify({
            'error': 'Both signature images must be uploaded',
            'match': False
        }), 400

    original_file = request.files['original_signature']
    verification_file = request.files['verification_signature']

    # Save uploaded files
    original_path = os.path.join(UPLOAD_FOLDER, 'original_signature.png')
    verification_path = os.path.join(UPLOAD_FOLDER, 'verification_signature.png')
    
    original_file.save(original_path)
    verification_file.save(verification_path)

    try:
        is_match, d = verify_signatures(original_path, verification_path)
        os.remove(original_path)
        os.remove(verification_path)
        return jsonify({
            'match': is_match,
            'confidence': float(d),
            'original_image': original_path,
            'verification_image': verification_path
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'match': False
        }), 500

if __name__ == '__main__':
    app.run(debug=True)