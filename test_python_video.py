import cv2
import time
from ultralytics import YOLO

print("=" * 60)
print("YOLO11 Python Test (Video File)")
print("=" * 60)

# Load model
print("\n[1] Loading YOLO11 model...")
model = YOLO("yolo11n.pt")
print(" Model loaded!")

# Open video file
print("\n[2] Opening test video...")
cap = cv2.VideoCapture("test_video.mp4")

if not cap.isOpened():
    print(" Cannot open test_video.mp4!")
    print("Make sure you ran record_test_video.py first!")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video opened! Total frames: {total_frames}")

# Process video
frame_count = 0
start_time = time.time()

print("\n[3] Processing video...")
print("Press 'q' to stop early")
print("-" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()
    
    frame_count += 1
    
    # Calculate FPS
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    
    # Display
    cv2.putText(annotated_frame, f"Python FPS: {fps:.2f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("YOLO11 Python - Video Test", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Final results
total_time = time.time() - start_time
final_fps = frame_count / total_time

print("-" * 60)
print("\n PYTHON RESULTS (Video File):")
print("=" * 60)
print(f"Frames processed: {frame_count}")
print(f"Total time: {total_time:.2f} seconds")
print(f" PYTHON FPS: {final_fps:.2f} ")
print("=" * 60)