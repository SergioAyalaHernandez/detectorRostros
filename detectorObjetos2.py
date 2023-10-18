import cv2
import dlib
import numpy as np
from mtcnn import MTCNN

# Inicialización de MTCNN y dlib
detector = MTCNN()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
font = cv2.FONT_HERSHEY_PLAIN


# Cargando YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

confidence_threshold = 0.5
nms_threshold = 0.4


def detectar_elementos(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    for i in range(len(boxes)):
        if i in indexes:
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 1, color, 2)

    return frame


def detectar_rostro(frame):
    faces = detector.detect_faces(frame)
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Usando dlib para marcar puntos faciales
        rect = dlib.rectangle(x, y, x + w, y + h)
        landmarks = predictor(frame, rect)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    return frame


cap = cv2.VideoCapture(1)  # Captura desde la cámara principal

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detectar_rostro(frame)
    frame = detectar_elementos(frame)

    cv2.imshow("Detector de Rostros y Objetos", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
