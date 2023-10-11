import requests
import os

# URLs para descargar los archivos necesarios
urls = {
    "yolov3.cfg": "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true",
    "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
    "coco.names": "https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true",
    #"yolo-glasses.cfg": "https://ruta/a/yolo-glasses.cfg",
    #"yolo-glasses.weights": "https://ruta/a/yolo-glasses.weights",
    #"glasses.names": "https://ruta/a/glasses.names"
}

for filename, url in urls.items():
    if not os.path.exists(filename):  # Verifica si el archivo ya existe
        print(f"Descargando {filename}...")
        response = requests.get(url, allow_redirects=True)

        with open(filename, 'wb') as file:
            file.write(response.content)
    else:
        print(f"{filename} ya existe, saltando la descarga.")

print("¡Descargas completadas!")
