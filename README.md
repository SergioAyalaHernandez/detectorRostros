# detectorRostros


Clonar Darknet (la implementación oficial de YOLO) en tu máquina:


git clone https://github.com/AlexeyAB/darknet.git
Compilar Darknet:

Ve al directorio clonado:


cd darknet
Edita el archivo Makefile para activar la compatibilidad con GPU (CUDA) y OpenCV, cambiando las líneas correspondientes a:


OPENCV=1
GPU=1
CUDNN=1
Compila Darknet:


make
Descargar los pesos y configuraciones de YOLOv3:

Puedes descargar los pesos oficiales de YOLOv3 con el siguiente comando:

wget https://pjreddie.com/media/files/yolov3.weights
Probar YOLO:

Una vez que hayas descargado los pesos y compilado Darknet, puedes probar YOLO en una imagen de prueba:

./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg


