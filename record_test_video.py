import cv2
import time

print("=" * 60)
print("Recording Test Video")
print("=" * 60)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam!")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30

print(f"\nRecording at {width}x{height}, {fps} FPS")
print("Recording will last 30 seconds...")
print("Move around, wave, hold up objects!")
print("\nStarting in 3 seconds...")
time.sleep(3)

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_video.mp4', fourcc, fps, (width, height))

start_time = time.time()
duration = 30  # Record for 30 seconds
frame_count = 0

print("\n RECORDING! Do interesting things for 30 seconds!")

while time.time() - start_time < duration:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Write frame to file
    out.write(frame)
    frame_count += 1
    
    # Show countdown
    remaining = int(duration - (time.time() - start_time))
    cv2.putText(frame, f"Recording: {remaining}s left", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Recording...", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

print("\n Recording complete!")
print(f"Saved: test_video.mp4")
print(f"Frames recorded: {frame_count}")
print(f"Duration: {duration} seconds")