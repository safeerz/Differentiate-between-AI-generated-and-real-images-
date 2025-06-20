from huggingface_hub import from_pretrained_keras

# Load the model
print("🔄 Loading model...")
model = from_pretrained_keras('Emmawang/mobilenet_v2_fake_image_detection')

# Print number of layers
print(f"\n✅ Number of layers in the model: {len(model.layers)}")

# Print full model summary
print("\n📊 Model Summary:")
model.summary()
