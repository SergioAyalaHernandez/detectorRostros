import os
import cv2
import face_recognition
import numpy as np
from collections import Counter
import logging
import json

logging.basicConfig(filename='detection_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(message)s')
# Cargar vectores de características preexistentes y etiquetas
base_dir = r"D:\OneDrive - AXEDE\capturas2"
personas = os.listdir(base_dir)
encodings_conocidos = []
etiquetas_conocidas = []
caracteristicas_conocidas = {}

for persona in personas:
    encodings_persona = [np.load(os.path.join(base_dir, persona, f)) for f in
                         os.listdir(os.path.join(base_dir, persona)) if f.endswith('.npy')]
    encodings_conocidos.extend(encodings_persona)
    etiquetas_conocidas.extend([persona] * len(encodings_persona))
    for file in os.listdir(os.path.join(base_dir, persona)):
        if file.startswith("caracteristicas_") and file.endswith('.json'):
            with open(os.path.join(base_dir, persona, file), 'r') as f:
                caracteristicas = json.load(f)
                caracteristicas_conocidas[persona] = caracteristicas
# Inicializar las cámaras (asumiendo 3 cámaras)
cams = [cv2.VideoCapture(i) for i in range(3)]


def detectar_por_caracteristicas(frame, face_location):
    # Implementa la detección basada en características adicionales aquí.
    # Si bien estamos usando solo dos características (gafas y color de camisa) por simplicidad,
    # puedes agregar más características según lo necesites.
    caracteristicas_frame = obtener_caracteristicas_adicionales(frame, face_location)

    for persona, caracteristicas in caracteristicas_conocidas.items():
        if caracteristicas['gafas'] == caracteristicas_frame['gafas'] and \
                np.allclose(np.array(caracteristicas['color_camisa']), np.array(caracteristicas_frame['color_camisa']),
                            atol=40):
            return persona
    return "Desconocido"

while True:
    for index, cam in enumerate(cams):
        ret, frame = cam.read()
        if not ret:
            continue

        # Detectar rostros y obtener encodings
        locs_rostros = face_recognition.face_locations(frame)
        encodings_rostros = face_recognition.face_encodings(frame, locs_rostros)

        for (top, right, bottom, left), encoding_rostro in zip(locs_rostros, encodings_rostros):
            matches = face_recognition.compare_faces(encodings_conocidos, encoding_rostro)
            nombre = "Desconocido"

            # Determinar el nombre basado en la mayor cantidad de coincidencias
            if True in matches:
                matched_labels = [etiquetas_conocidas[i] for i, match in enumerate(matches) if match]
                nombre = Counter(matched_labels).most_common(1)[0][0]
                logging.info(f"Persona detectada por rostro: {nombre}")
            else:
                # Intentar detectar por características adicionales
                nombre = detectar_por_caracteristicas(frame, (top, right, bottom, left))
                if nombre != "Desconocido":
                    logging.info(f"Persona detectada por características adicionales: {nombre}")

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, nombre, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow(f'Camara {index}', frame)

    # Terminar el bucle con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cam in cams:
    cam.release()

cv2.destroyAllWindows()
