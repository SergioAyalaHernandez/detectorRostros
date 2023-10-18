import cv2
import os
import numpy as np

direccion = 'D:/accesoriosRostro'
lista = os.listdir(direccion)

etiquetas = []
rostros = []
con = 0

for nameDir in lista:
    nombre = direccion + '/' + nameDir

    for fileName in os.listdir(nombre):
        etiquetas.append(con)
        rostros.append(cv2.imread(nombre + '/' + fileName, 0))

    con = con + 1

reconocimiento = cv2.face.LBPHFaceRecognizer_create()

reconocimiento.train(rostros, np.array(etiquetas))

reconocimiento.write('modeloAccesoriosGafas.xml')
print('Modelo creado')
