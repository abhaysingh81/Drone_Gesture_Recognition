import cv2
import os


LABELS = {
    0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
    5: "Five", 6: "Fist", 7: "Claw", 8: "ThumbsUp", 9: "Peace"
}

DATA_DIR = "gesture_data"
os.makedirs(DATA_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    
    # Display instructions on screen
    cv2.putText(frame, "Press 0-9 to Capture | 'q' to Quit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Data Collector', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Check if a number key was pressed (ASCII 48 is '0', 57 is '9')
    if 48 <= key <= 57:
        class_idx = key - 48
        class_name = LABELS[class_idx]
        
        # Count existing files to create unique name
        count = len([f for f in os.listdir(DATA_DIR) if f.startswith(f"{class_idx}_")])
        filename = f"{DATA_DIR}/{class_idx}_{class_name}_{count}.jpg"
        
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        
        # Flash effect
        cv2.imshow('Data Collector', 255 - frame)
        cv2.waitKey(50)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()