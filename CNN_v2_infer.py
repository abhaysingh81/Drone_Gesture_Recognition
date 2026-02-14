from CNN_v2 import myCNN
from PIL import Image
import torch
import cv2
from torchvision import transforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = {
    0: "Move_Forward, nice", 1: "Move_Backward, one", 2: "Left, two", 3: "Right, three", 4: "Stop, four",
    5: "Move_Up, five", 6: "Move_Down, fist", 7: "Land, claw", 8: "Spin_Clk, thumbsUp", 9: "Spin_AntiClk, peace"
}

def inference(model):
    cap = cv2.VideoCapture(0)
    IMG_SIZE = 128
    
    inference_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    model.eval()

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        roi_size = 300
        x1, y1 = int(w/2 - roi_size/2), int(h/2 - roi_size/2)
        x2, y2 = x1 + roi_size, y1 + roi_size
     
        roi = frame[y1:y2, x1:x2]
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        roi_tensor = inference_transform(roi_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(roi_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
        class_id = predicted.item()
        class_label = LABELS[class_id]
        conf_score = confidence.item()

        color = (0, 255, 0) if conf_score > 0.7 else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        text = f"{class_label} ({conf_score*100:.1f}%)"
        print (text)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow("Hand Gesture Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model = myCNN(num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load("hand_gesture_cnn.pth", map_location=DEVICE))
    inference(model)