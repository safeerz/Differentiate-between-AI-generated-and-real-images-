import tensorflow as tf
import os

# 1. Load your existing model (replace with your actual model path)
model = tf.keras.models.load_model("model.h5")
print("Original model loaded.")

# 2. Save the model in the new Keras format (.keras)
model.save("model.keras", save_format="keras")
print("Model saved as 'model.keras'")

# 3. Load the saved model to verify
loaded_model = tf.keras.models.load_model("model.keras")
print("Model loaded successfully from 'model.keras'")

import tensorflow as tf

model = tf.keras.models.load_model("model.keras")
print(model.summary())
