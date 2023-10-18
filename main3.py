import cv2
import os
import dlib
import numpy as np
import json
from mtcnn import MTCNN
import face_recognition

# Inicializa MTCNN y dlib
detector = MTCNN()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


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


def obtener_caracteristicas_adicionales(frame, face_location):
    ''' Esta función detectará características adicionales en la imagen como gafas y el color de la camisa.
        Devolverá un diccionario con estas características.
    '''
    caracteristicas = {}

    # Detección de gafas usando Haar cascades de OpenCV
    glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    glasses = glasses_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
    if len(glasses) > 0:
        caracteristicas['gafas'] = True
    else:
        caracteristicas['gafas'] = False

    # Detección de color de la camisa basado en el color promedio debajo de la cara
    top, right, bottom, left = face_location
    shirt_region = frame[bottom:bottom + (bottom - top) // 2, left:right]
    mean_color = cv2.mean(shirt_region)[:3]
    caracteristicas['color_camisa'] = mean_color

    return caracteristicas

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
            ret_main, frame_main = cams[2].read()
            if not ret_main:
                continue

            # Detección de rostros con MTCNN
            faces = detector.detect_faces(frame_main)
            for face in faces:
                x, y, w, h = face['box']
                rect = dlib.rectangle(x, y, x + w, y + h)
                landmarks = predictor(frame_main, rect)

                top = y
                right = x + w
                bottom = y + h
                left = x

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

                        # Detección y guardado de características adicionales
                        caracteristicas = obtener_caracteristicas_adicionales(frame, (top, right, bottom, left))
                        caracteristicas_filename = os.path.join(dir_path, f"caracteristicas_{accion}_cam_{j}.json")
                        with open(caracteristicas_filename, 'w') as f:
                            json.dump(caracteristicas, f)

                break

            cv2.putText(frame_main, accion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Camara Principal', frame_main)

            key = cv2.waitKey(1)
            if key == ord('q'):
                for cam in cams:
                    cam.release()
                cv2.destroyAllWindows()
                return

    for cam in cams:
        cam.release()
    cv2.destroyAllWindows()


nombre_persona = input("Introduce el nombre de la persona: ")
capturar_rostros(nombre_persona)
