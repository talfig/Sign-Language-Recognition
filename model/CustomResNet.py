from tensorflow.keras.applications import ResNet50

# Load the ResNet50 model with pre-trained weights
model = ResNet50(weights='imagenet')

# Print the model summary
model.summary()