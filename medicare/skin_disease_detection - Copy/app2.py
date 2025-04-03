# # Import necessary libraries
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.models import Model

# # Step 1: Data Preparation
# # Path to your dataset
# data_path = r"D:\\skin_disease_detection\\dataset"  # Replace with your actual dataset path

# # Image data generator with data augmentation
# train_datagen = ImageDataGenerator(
#     rescale=1.0/255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     validation_split=0.2  # Reserve 20% of data for validation
# )

# # Training data generator
# train_generator = train_datagen.flow_from_directory(
#     data_path,
#     target_size=(128, 128),  # Resize images to 128x128
#     batch_size=32,
#     class_mode='categorical',
#     subset='training'  # 80% of data for training
# )

# # Validation data generator
# validation_generator = train_datagen.flow_from_directory(
#     data_path,
#     target_size=(128, 128),
#     batch_size=32,
#     class_mode='categorical',
#     subset='validation'  # 20% of data for validation
# )

# # Step 2: Define the Model
# # Load MobileNetV2 with pre-trained weights, exclude the top layers
# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
# base_model.trainable = False  # Freeze the base model layers

# # Add custom layers on top
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(128, activation='relu')(x)
# x = Dense(train_generator.num_classes, activation='softmax')(x)  # Output layer with 8 classes

# model = Model(inputs=base_model.input, outputs=x)

# # Step 3: Compile the Model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Step 4: Train the Model
# history = model.fit(
#     train_generator,
#     epochs=10,  # Adjust epochs based on dataset size
#     validation_data=validation_generator
# )

# # Step 5: Save the Model
# model.save('skin_disease_model.h5')
# print("Model saved as skin_disease_model.h5")

# # Optional: Evaluate the Model
# loss, accuracy = model.evaluate(validation_generator)
# print(f"Validation Accuracy: {accuracy * 100:.2f}%")

from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Flask server is running!"

if __name__ == "_main_":
    app.run(debug=True)