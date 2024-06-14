import cv2
from ultralytics import YOLO

# Carica il modello YOLO
detector = YOLO(r"C:\path\best.pt")

video_path = r"C:\path\video_test.mp4"
cap = cv2.VideoCapture(video_path)

output_path = r"C:\path\video_out.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Effettua il rilevamento
    results = detector(frame)         #results = 1 perchè agisce su un singolo frame per volta
    detections = results[0].boxes     #detections contiene la lista di bounding boxes detected

    for box in detections:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        cls = results[0].names[box.cls[0].item()]   

        # Estrai la regione di interesse (ROI)
        roi = frame[int(y1):int(y2), int(x1):int(x2)]

        # Aggiungi informazioni sul frame
        label = f'Class: {cls}, Conf: {conf:.2f}'
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Scrivi il frame nel video di output
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
