# Initialize model
model = TTSModel()
print("Model structure created successfully!")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
