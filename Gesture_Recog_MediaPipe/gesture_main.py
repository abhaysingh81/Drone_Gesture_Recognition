import cv2
import os
import mediapipe as mp
#from mediapipe.tasks import python
#from mediapipe.tasks.python import vision

"""7 hand gestures:
Closed_Fist

Open_Palm

Pointing_Up

Thumb_Down

Thumb_Up

Victory

ILoveYou """

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'gesture_recognizer.task')

base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
options = mp.tasks.vision.GestureRecognizerOptions(base_options=base_options)
recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    results = recognizer.recognize(mp_image)

    if results.gestures:
        top_gesture = results.gestures[0][0].category_name
        if top_gesture == "Closed_Fist":
            print("Drone - Land")
        elif top_gesture == "Open_Palm":
            print("Drone - Takeoff")
        elif top_gesture == "Pointing_Up":
            print("Drone - Move Forward")
        elif top_gesture == "Thumb_Down":
            print("Drone - Move Backward")
        elif top_gesture == "Thumb_Up":
            print("Drone - Move Up")
        elif top_gesture == "Thumb_Down":
            print("Drone - Move Down")
        elif top_gesture == "Victory":
            print("Drone - Rotate(Clockwise)")
        elif top_gesture == "ILoveYou":
            print("Drone - Rotate(Counter-Clockwise)")
            
        cv2.putText(frame, f'Gesture: {top_gesture}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('7-Gesture Classifier', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()