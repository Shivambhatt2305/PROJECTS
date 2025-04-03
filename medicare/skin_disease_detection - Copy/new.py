import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

# Step 1: Dataset Paths and Model Path
data_path = "D:\\skin_disease_detection\\dataset"  # Update to your dataset path
model_path = "skin_disease_model.h5"

# Step 2: Define Labels and Treatments
labels = ["cellulitis", "impetigo", "athlete's foot", "nail fungus", "ringworm", 
          "cutaneous larva migrans", "chickenpox", "shingles"]

treatment_suggestions = {
    "cellulitis": "Treatment includes antibiotics. | સારવારમાં એન્ટીબાયોટિક્સનો સમાવેશ થાય છે.",
    "impetigo": "Topical antibiotics are recommended. | ટોપિકલ એન્ટીબાયોટિક્સ ભલામણ કરવામાં આવે છે.",
    "athlete's foot": "Use antifungal creams and keep feet dry. | એન્ટીફંગલ ક્રિમનો ઉપયોગ કરો અને પગને સુકા રાખો.",
    "nail fungus": "Oral antifungal medication is effective. | ઓરલ એન્ટીફંગલ દવાઓ અસરકારક છે.",
    "ringworm": "Antifungal cream for affected area. | અસરગ્રસ્ત વિસ્તારમાં એન્ટીફંગલ ક્રિમ લાગુ કરો.",
    "cutaneous larva migrans": "Oral antiparasitic drugs are suggested. | ઓરલ એન્ટીપેરાસિટિક દવાઓની ભલામણ કરવામાં આવે છે.",
    "chickenpox": "Treatment includes antiviral medication and rest. | સારવારમાં એન્ટીવાયરલ દવાઓ અને આરામનો સમાવેશ થાય છે.",
    "shingles": "Antiviral drugs and pain relief recommended. | એન્ટીવાયરલ દવાઓ અને પીડા નિવારક ભલામણ કરવામાં આવે છે."
}

# Step 3: Data Generators
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)
validation_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Step 4: Build Model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(len(labels), activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Step 5: Train Model
model.fit(train_generator, epochs=10, validation_data=validation_generator)
model.save(model_path)

# Step 6: Evaluate Model
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Step 7: Real-Time Detection with Treatment Suggestions
# Load the trained model
model = load_model(model_path)

# Open webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0) / 255.0

    # Prediction
    predictions = model.predict(img)
    class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    label = labels[class_index]
    treatment = treatment_suggestions.get(label, "Consult a healthcare provider for advice. | સલાહ માટે આરોગ્ય સંભાળ પ્રદાતા સાથે પરામર્શ કરો.")

    # Display results
    cv2.putText(frame, f"{label}: {confidence:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Treatment: {treatment}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Skin Disease Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
