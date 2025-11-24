import sys
import os
if '--generate-requirements' in sys.argv:
    requirements = """Flask==3.0.0
flask-cors==4.0.0
tensorflow==2.15.0
Pillow==10.1.0
numpy==1.24.3
gunicorn==21.2.0"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    print("✅ requirements.txt generated successfully!")
    print("Now push both app.py and requirements.txt to GitHub and deploy on Render")
    sys.exit(0)

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import base64
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load pretrained MobileNetV2 model
print("Loading MobileNetV2 model...")
model = MobileNetV2(weights='imagenet')
print("Model loaded successfully!")

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Image Classifier</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
            max-width: 600px;
            width: 100%;
        }
        h1 {
            text-align: center;
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 20px;
        }
        .upload-area:hover {
            background: #f8f9ff;
            border-color: #764ba2;
        }
        .upload-area.dragover {
            background: #e8ebff;
            border-color: #764ba2;
        }
        input[type="file"] {
            display: none;
        }
        .upload-icon {
            font-size: 50px;
            margin-bottom: 10px;
        }
        .upload-text {
            color: #667eea;
            font-size: 1.2em;
            font-weight: 600;
        }
        #preview {
            max-width: 100%;
            max-height: 300px;
            border-radius: 10px;
            margin: 20px auto;
            display: none;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        #results {
            margin-top: 20px;
            display: none;
        }
        .result-item {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            animation: slideIn 0.5s;
        }
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        .result-label {
            font-weight: 600;
            font-size: 1.1em;
        }
        .result-confidence {
            background: rgba(255,255,255,0.3);
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: 600;
        }
        .loading {
            text-align: center;
            color: #667eea;
            font-size: 1.2em;
            display: none;
            margin: 20px 0;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-size: 1em;
            cursor: pointer;
            transition: transform 0.2s;
            margin-top: 10px;
        }
        .btn:hover {
            transform: scale(1.05);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI Image Classifier</h1>
        <p class="subtitle">Upload an image and let AI identify it!</p>
        
        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">📸</div>
            <div class="upload-text">Click or drag image here</div>
            <input type="file" id="fileInput" accept="image/*">
        </div>
        
        <img id="preview" alt="Preview">
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            Analyzing image...
        </div>
        
        <div id="results"></div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const preview = document.getElementById('preview');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');

        uploadArea.addEventListener('click', () => fileInput.click());
        
        fileInput.addEventListener('change', handleFile);
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                fileInput.files = e.dataTransfer.files;
                handleFile();
            }
        });

        async function handleFile() {
            const file = fileInput.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = async (e) => {
                preview.src = e.target.result;
                preview.style.display = 'block';
                results.style.display = 'none';
                loading.style.display = 'block';

                const formData = new FormData();
                formData.append('file', file);

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    loading.style.display = 'none';
                    displayResults(data.predictions);
                } catch (error) {
                    loading.style.display = 'none';
                    alert('Error: ' + error.message);
                }
            };
            reader.readAsDataURL(file);
        }

        function displayResults(predictions) {
            results.innerHTML = '<h2 style="color: #667eea; margin-bottom: 15px;">Top Predictions:</h2>';
            predictions.forEach((pred, index) => {
                results.innerHTML += `
                    <div class="result-item" style="animation-delay: ${index * 0.1}s">
                        <span class="result-label">${pred.label}</span>
                        <span class="result-confidence">${pred.confidence}</span>
                    </div>
                `;
            });
            results.style.display = 'block';
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and preprocess image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Make prediction
        predictions = model.predict(img_array)
        decoded = decode_predictions(predictions, top=5)[0]
        
        # Format results
        results = [
            {
                'label': label.replace('_', ' ').title(),
                'confidence': f'{score * 100:.2f}%'
            }
            for (_, label, score) in decoded
        ]
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)