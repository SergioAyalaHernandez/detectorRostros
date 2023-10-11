import requests

# URLs para descargar los archivos necesarios
urls = {
    "yolov3.cfg": "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true",
    "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
    "coco.names": "https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true"
}

for filename, url in urls.items():
    print(f"Descargando {filename}...")
    response = requests.get(url, allow_redirects=True)

    with open(filename, 'wb') as file:
        file.write(response.content)

print("¡Descargas completadas!")
