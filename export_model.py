from ultralytics import YOLO

print("=" * 50)
print("YOLO11 Model Export Script")
print("=" * 50)

# Download and load YOLO11 Nano model
print("\n[Step 1] Loading YOLO11 Nano model...")
print("(This will download ~6MB on first run)")
model = YOLO("yolo11n.pt")
print("Model loaded successfully!")

# Export to ONNX format
print("\n[Step 2] Exporting to ONNX format...")
print("(This takes 30-60 seconds)")
model.export(format="onnx", dynamic=True, simplify=True)
print("Export complete!")

print("\n" + "=" * 50)
print("SUCCESS! Check your folder for 'yolo11n.onnx'")
print("=" * 50)