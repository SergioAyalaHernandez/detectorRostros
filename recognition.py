import os
import cv2
import face_recognition
import numpy as np

# Cargar vectores de características preexistentes y etiquetas
base_dir = r"D:\OneDrive - AXEDE\capturas2"
personas = os.listdir(base_dir)
encodings_conocidos = []
etiquetas_conocidas = []

for persona in personas:
    encodings_persona = [np.load(os.path.join(base_dir, persona, f)) for f in
                         os.listdir(os.path.join(base_dir, persona)) if f.endswith('.npy')]
    encodings_conocidos.extend(encodings_persona)
    etiquetas_conocidas.extend([persona] * len(encodings_persona))

# Inicializar las cámaras (asumiendo 3 cámaras)
cams = [cv2.VideoCapture(i) for i in range(3)]

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

            if True in matches:
                indice_best_match = matches.index(True)
                nombre = etiquetas_conocidas[indice_best_match]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, nombre, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow(f'Camara {index}', frame)

    # Terminar el bucle con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cam in cams:
    cam.release()

cv2.destroyAllWindows()
