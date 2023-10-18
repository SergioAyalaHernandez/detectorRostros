import cv2
import os
import dlib
import numpy as np
import json
from mtcnn import MTCNN
import face_recognition

def json_serial(obj):
    """Función para serializar tipos de datos no serializables por defecto."""
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


# Inicializa MTCNN y dlib
detector = MTCNN()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Configuración para YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
unconnected_layers = net.getUnconnectedOutLayers().flatten()
output_layers = [layer_names[i - 1] for i in unconnected_layers]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def detect_objects(img):
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    detected_objects = []

    for out in detections:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                detected_objects.append({"class": classes[class_id], "confidence": confidence})

                center_x, center_y, w, h = map(int, detection[0:4] * [width, height, width, height])
                top_left_x = int(center_x - w / 2)
                top_left_y = int(center_y - h / 2)
                cv2.rectangle(img, (top_left_x, top_left_y), (top_left_x + w, top_left_y + h), (0, 255, 0), 2)
                cv2.putText(img, classes[class_id], (top_left_x, top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return detected_objects

def esta_sonriendo(landmarks):
    top_lip_center = (landmarks.part(51).x, landmarks.part(51).y)
    bottom_lip_center = (landmarks.part(57).x, landmarks.part(57).y)
    lip_distance = abs(top_lip_center[1] - bottom_lip_center[1])
    return lip_distance > 20


def ratio_ojo(puntos, landmarks):
    A = ((landmarks.part(puntos[1]).x - landmarks.part(puntos[5]).x) ** 2 +
         (landmarks.part(puntos[1]).y - landmarks.part(puntos[5]).y) ** 2) ** 0.5
    B = ((landmarks.part(puntos[2]).x - landmarks.part(puntos[4]).x) ** 2 +
         (landmarks.part(puntos[2]).y - landmarks.part(puntos[4]).y) ** 2) ** 0.5
    C = ((landmarks.part(puntos[0]).x - landmarks.part(puntos[3]).x) ** 2 +
         (landmarks.part(puntos[0]).y - landmarks.part(puntos[3]).y) ** 2) ** 0.5
    ratio = (A + B) / (2.0 * C)
    return ratio


def mirando_izquierda(landmarks):
    distancia_ojos = ((landmarks.part(39).x - landmarks.part(42).x) ** 2 +
                      (landmarks.part(39).y - landmarks.part(42).y) ** 2) ** 0.5
    distancia_nariz_ojo_derecho = ((landmarks.part(33).x - landmarks.part(42).x) ** 2 +
                                   (landmarks.part(33).y - landmarks.part(42).y) ** 2) ** 0.5
    return distancia_ojos < distancia_nariz_ojo_derecho * 0.6


def mirando_derecha(landmarks):
    distancia_ojos = ((landmarks.part(42).x - landmarks.part(39).x) ** 2 +
                      (landmarks.part(42).y - landmarks.part(39).y) ** 2) ** 0.5
    distancia_nariz_ojo_izquierdo = ((landmarks.part(33).x - landmarks.part(39).x) ** 2 +
                                     (landmarks.part(33).y - landmarks.part(39).y) ** 2) ** 0.5
    return distancia_ojos < distancia_nariz_ojo_izquierdo * 0.6

def capturar_rostros(nombre_persona, num_fotos=5):
    base_dir = r"D:\OneDrive - AXEDE\capturas2"
    dir_path = os.path.join(base_dir, nombre_persona)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cams = [cv2.VideoCapture(i) for i in range(3)]
    acciones = ["Sonríe", "Parpadea", "Mira a la izquierda", "Mira a la derecha", "Cara neutral"]

    for accion in acciones:
        accion_detectada = False

        while not accion_detectada:
            ret_main, frame_main = cams[1].read()
            if not ret_main:
                continue

            detected_objects = detect_objects(frame_main)
            faces = detector.detect_faces(frame_main)

            for face in faces:
                x, y, w, h = face['box']
                rect = dlib.rectangle(x, y, x + w, y + h)
                landmarks = predictor(frame_main, rect)

                # Revisar las acciones y actualizar el estado accion_detectada
                if accion == "Sonríe" and esta_sonriendo(landmarks):
                    accion_detectada = True
                elif accion == "Parpadea" and (ratio_ojo([36, 37, 38, 39, 40, 41], landmarks) < 0.25 or
                                               ratio_ojo([42, 43, 44, 45, 46, 47], landmarks) < 0.25):
                    accion_detectada = True
                elif accion == "Mira a la izquierda" and mirando_izquierda(landmarks):
                    accion_detectada = True
                elif accion == "Mira a la derecha" and mirando_derecha(landmarks):
                    accion_detectada = True
                elif accion == "Cara neutral":
                    accion_detectada = True

                if accion_detectada:
                    for j, cam in enumerate(cams):
                        ret, frame = cam.read()
                        if ret:
                            filename = os.path.join(dir_path, f"foto_{accion}_cam_{j}.jpg")
                            cv2.imwrite(filename, frame)

                            # Obtener el vector de características del rostro
                            face_encodings = face_recognition.face_encodings(frame)
                            if face_encodings:
                                face_encoding = face_encodings[0]
                                encoding_filename = os.path.join(dir_path, f"encoding_{accion}_cam_{j}.npy")
                                np.save(encoding_filename, face_encoding)
                    break  # Salir del bucle for interno una vez que se ha detectado la acción y se han guardado las imágenes

            cv2.putText(frame_main, accion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Camara Principal', frame_main)

            key = cv2.waitKey(1)
            if key == ord('q'):
                for cam in cams:
                    cam.release()
                cv2.destroyAllWindows()
                return


        # Guardamos la información de los objetos detectados en un archivo JSON
        with open(os.path.join(dir_path, f"detections_{accion}.json"), "w") as json_file:
            json.dump(detected_objects, json_file, default=json_serial)

    for cam in cams:
        cam.release()
    cv2.destroyAllWindows()

nombre_persona = input("Introduce el nombre de la persona: ")
capturar_rostros(nombre_persona)
