import cv2
import time
from ultralytics import YOLO

print("=" * 60)
print("YOLO11 Python Baseline Performance Test")
print("=" * 60)

# Load the model
print("\n[1] Loading YOLO11 model...")
model = YOLO("yolo11n.pt")
print(" Model loaded!")

# Try to open webcam (change to video file if no webcam)
print("\n[2] Opening webcam...")
print("(Press 'q' to quit)")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" No webcam found!")
    print(" Tip: Change line 'cap = cv2.VideoCapture(0)' to")
    print("         'cap = cv2.VideoCapture(\"your_video.mp4\")'")
    exit()

print(" Webcam opened!")

# FPS tracking
frame_count = 0
start_time = time.time()
fps_display = 0

print("\n[3] Starting inference...")
print("Watch the video window. Press 'q' to stop.")
print("-" * 60)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print(" Failed to grab frame")
        break
    
    # Run YOLO detection (THE MAIN ACTION!)
    results = model(frame, verbose=False)
    
    # Draw bounding boxes
    annotated_frame = results[0].plot()
    
    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    
    # Update FPS every 10 frames
    if frame_count % 10 == 0:
        fps_display = frame_count / elapsed_time
    
    # Draw FPS text on the video
    cv2.putText(annotated_frame, f"Python FPS: {fps_display:.1f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the video window
    cv2.imshow("YOLO11 - Python Baseline", annotated_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Final results
final_fps = frame_count / elapsed_time

print("-" * 60)
print("\n FINAL RESULTS:")
print("=" * 60)
print(f"Total frames processed: {frame_count}")
print(f"Total time: {elapsed_time:.2f} seconds")
print(f" BASELINE PYTHON FPS: {final_fps:.2f} ")
print("=" * 60)
print("\n WRITE THIS NUMBER DOWN!")
print("This is your 'before' speed that we'll improve with C++")
print("=" * 60)