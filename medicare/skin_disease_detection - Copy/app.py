from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
import numpy as np
import cv2
import base64
import os
import uuid
from datetime import datetime

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Labels and Treatments
labels = ["cellulitis", "impetigo", "athlete's foot", "nail fungus", "ringworm", 
          "cutaneous larva migrans", "chickenpox", "shingles"]

treatment_suggestions = {
    "cellulitis": "Treatment includes antibiotics like Keflex or Bactrim. Keep the area clean and elevated.",
    "impetigo": "Topical antibiotics such as Mupirocin are recommended. For severe cases, oral antibiotics may be needed.",
    "athlete's foot": "Use antifungal creams like Clotrimazole or Terbinafine. Keep feet dry and change socks regularly.",
    "nail fungus": "Oral antifungal medication like Terbinafine or Itraconazole is effective. Treatment may take several months.",
    "ringworm": "Antifungal cream such as Miconazole or Clotrimazole for the affected area. Maintain good hygiene.",
    "cutaneous larva migrans": "Oral antiparasitic drugs like Albendazole or Ivermectin are suggested. Avoid walking barefoot on contaminated soil.",
    "chickenpox": "Treatment includes antiviral medication like Acyclovir, calamine lotion for itching, and rest. Stay hydrated.",
    "shingles": "Antiviral drugs like Valacyclovir and pain relief medications are recommended. Early treatment is important."
}

disease_descriptions = {
    "cellulitis": "Cellulitis is a common bacterial skin infection that causes redness, swelling, and pain in the infected area. It occurs when bacteria enter through a break in the skin.",
    "impetigo": "Impetigo is a highly contagious bacterial skin infection that causes red sores that quickly rupture, ooze for a few days, and then form a yellowish-brown crust.",
    "athlete's foot": "Athlete's foot is a fungal infection that usually begins between the toes. It commonly occurs in people whose feet have become very sweaty while confined within tight-fitting shoes.",
    "nail fungus": "Nail fungus is a common condition that begins as a white or yellow spot under the tip of your fingernail or toenail. As the fungal infection goes deeper, it may cause your nail to discolor, thicken and crumble at the edge.",
    "ringworm": "Ringworm is a common fungal infection that can cause a red or silvery ring-like rash on the skin. Despite its name, it's not caused by a worm but by fungi living off the dead tissue of your skin, hair, and nails.",
    "cutaneous larva migrans": "Cutaneous larva migrans is a skin disease caused by hookworm larvae that usually infect cats and dogs. Humans can be infected when walking barefoot on contaminated soil or sand.",
    "chickenpox": "Chickenpox is a highly contagious disease caused by the varicella-zoster virus. It causes an itchy, blister-like rash that appears first on the chest, back, and face, and then spreads over the entire body.",
    "shingles": "Shingles is a viral infection that causes a painful rash. It is caused by the varicella-zoster virus, the same virus that causes chickenpox. After you've had chickenpox, the virus lies inactive in nerve tissue and may reactivate as shingles years later."
}

# Load the model
model_path = "skin_disease_model_v2.h5"
model = None

def load_model():
    global model
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/detect', methods=['POST'])
def detect_disease():
    if model is None and not load_model():
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Get image data from request
        data = request.json
        image_data = data['image']
        
        # Remove header from base64 string
        if 'data:image' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Save the image
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, img)
        
        # Preprocess the image
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0) / 255.0
        
        # Make prediction
        predictions = model.predict(img)
        confidence = float(np.max(predictions[0]) * 100)
        class_index = np.argmax(predictions[0])
        disease = labels[class_index]
        
        # Get treatment and description
        treatment = treatment_suggestions.get(disease, "Consult a healthcare provider for advice.")
        description = disease_descriptions.get(disease, "No detailed information available.")
        
        # Return results
        return jsonify({
            "disease": disease.title(),
            "confidence": round(confidence, 2),
            "description": description,
            "treatments": [treatment],
            "image_path": f"/static/uploads/{filename}"
        })
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_image():
    if model is None and not load_model():
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save the file
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read and preprocess the image
        img = cv2.imread(filepath)
        img_resized = cv2.resize(img, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0) / 255.0
        
        # Make prediction
        predictions = model.predict(img_array)
        confidence = float(np.max(predictions[0]) * 100)
        class_index = np.argmax(predictions[0])
        disease = labels[class_index]
        
        # Get treatment and description
        treatment = treatment_suggestions.get(disease, "Consult a healthcare provider for advice.")
        description = disease_descriptions.get(disease, "No detailed information available.")
        
        # Return results
        return jsonify({
            "disease": disease.title(),
            "confidence": round(confidence, 2),
            "description": description,
            "treatments": [treatment],
            "image_path": f"/static/uploads/{filename}"
        })
    
    except Exception as e:
        print(f"Error processing uploaded image: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Try to load the model at startup
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
