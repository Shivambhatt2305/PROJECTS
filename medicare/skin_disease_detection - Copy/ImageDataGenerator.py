from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Use 20% of data for validation
)

# Load training images
train_generator = train_datagen.flow_from_directory(
    'path_to_dataset',  # Replace with the actual dataset path
    target_size=(128, 128),  # Resize images to match input size of the model
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Load validation images
validation_generator = train_datagen.flow_from_directory(
    'path_to_dataset',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
