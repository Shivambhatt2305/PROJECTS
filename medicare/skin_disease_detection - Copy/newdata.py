# import tensorflow as tf
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import cv2
# import numpy as np
# import os

# # Step 1: Dataset Paths and Model Path
# data_path = "new dataset"  # Ensure this path is correct and accessible
# model_path = "skin_disease_model.h5"

# # Step 2: Define Labels and Treatments
# labels = ["dermatitis", "pimples", "vitiligo", "scarlet fever"]
# treatment_suggestions = {
#     "dermatitis": "Apply topical corticosteroids and moisturizers.",
#     "pimples": "Use salicylic acid or benzoyl peroxide products.",
#     "vitiligo": "Treatment includes phototherapy or topical corticosteroids.",
#     "scarlet fever": "Antibiotics like penicillin are recommended."
# }

# # Verify dataset structure
# if not os.path.exists(data_path):
#     raise FileNotFoundError(f"Dataset path '{data_path}' not found!")
# for label in labels:
#     if not os.path.exists(os.path.join(data_path, label)):
#         raise FileNotFoundError(f"Subfolder '{label}' is missing in the dataset!")

# # Step 3: Data Generators
# train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

# train_generator = train_datagen.flow_from_directory(
#     data_path,
#     target_size=(224, 224),
#     batch_size=8,  # Reduce batch size for debugging purposes
#     class_mode="categorical",
#     subset="training",
#     shuffle=True
# )

# validation_generator = train_datagen.flow_from_directory(
#     data_path,
#     target_size=(224, 224),
#     batch_size=8,
#     class_mode="categorical",
#     subset="validation",
#     shuffle=True
# )

# # Step 4: Build Model
# model = Sequential([
#     Input(shape=(224, 224, 3)),
#     Conv2D(32, (3, 3), activation="relu"),
#     MaxPooling2D(2, 2),
#     Conv2D(64, (3, 3), activation="relu"),
#     MaxPooling2D(2, 2),
#     Conv2D(128, (3, 3), activation="relu"),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(128, activation="relu"),
#     Dense(len(labels), activation="softmax")
# ])
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# # Step 5: Train Model
# try:
#     model.fit(train_generator, epochs=10, validation_data=validation_generator)
#     model.save(model_path)
# except ValueError as e:
#     print(f"Error during training: {e}")
#     print("Ensure dataset is properly configured and contains enough images for all classes.")

# # Step 6: Evaluate Model
# try:
#     val_loss, val_accuracy = model.evaluate(validation_generator)
#     print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
# except Exception as e:
#     print(f"Error during evaluation: {e}")

# # Step 7: Real-Time Detection
# model = load_model(model_path)

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     img = cv2.resize(frame, (224, 224))
#     img = np.expand_dims(img, axis=0) / 255.0

#     predictions = model.predict(img)
#     class_index = np.argmax(predictions[0])
#     confidence = np.max(predictions[0])
#     if confidence > 0.6:
#         label = labels[class_index]
#         treatment = treatment_suggestions.get(label, "N/A")
#     else:
#         label = "No disease detected"
#         treatment = "N/A"

#     cv2.putText(frame, f"{label}: {confidence:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.putText(frame, f"Treatment: {treatment}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#     cv2.imshow("Skin Disease Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the model
model_path = "skin_disease_model.h5"
model = load_model(model_path)

# Define labels and treatment suggestions
labels = ["dermatitis", "pimples", "vitiligo", "scarlet fever"]
treatment_suggestions = {
    "dermatitis": "Apply topical corticosteroids and moisturizers.",
    "pimples": "Use salicylic acid or benzoyl peroxide products.",
    "vitiligo": "Treatment includes phototherapy or topical corticosteroids.",
    "scarlet fever": "Antibiotics like penicillin are recommended."
}

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is accessible
if not cap.isOpened():
    print("Error: Could not open webcam.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Resize target for faster processing
frame_width, frame_height = 224, 224  # Ensure the size matches the model input

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture image from camera.")
        break

    # Resize the image to match model input shape
    img = cv2.resize(frame, (frame_width, frame_height))
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize the image
    
    try:
        predictions = model.predict(img)
        print(f"Predictions shape: {predictions.shape}")  # Debugging: Check predictions shape
        class_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Check if class_index is within the valid range
        if class_index >= len(labels):
            print(f"Invalid class index: {class_index}. Predictions: {predictions}")
            label = "Unknown"
            treatment = "N/A"
        else:
            label = labels[class_index]
            treatment = treatment_suggestions.get(label, "N/A")
    except Exception as e:
        print(f"Error during prediction: {e}")
        continue

    # Display the result
    cv2.putText(frame, f"{label}: {confidence:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Treatment: {treatment}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Skin Disease Detection", frame)

    # Handle key press and exit
    key = cv2.waitKey(1)  # Wait for 1ms to catch the key press
    if key == ord('q') or key == ord('Q'):  # Check for both lowercase and uppercase 'q'
        print("Exiting...")
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
