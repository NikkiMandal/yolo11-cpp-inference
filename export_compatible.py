from ultralytics import YOLO

print("=" * 60)
print("YOLO11 Compatible ONNX Export")
print("=" * 60)

# Load the model
print("\n[1] Loading YOLO11 Nano model...")
model = YOLO("yolo11n.pt")
print(" Model loaded!")

# Export with OLDER opset for compatibility
print("\n[2] Exporting to ONNX with opset 13...")
print("(This version is compatible with older ONNX Runtime)")
model.export(
    format="onnx",
    dynamic=False,      # Fixed input size for better compatibility
    simplify=True,
    opset=13           # Force older opset version
)
print(" Export complete!")

print("\n" + "=" * 60)
print("SUCCESS! The yolo11n.onnx file is now compatible!")
print("=" * 60)