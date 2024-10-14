import cv2

# Cargar el clasificador preentrenado de OpenCV para detección de rostros
cascade_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detectar_rostros(imagen):
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    rostros = cascade_rostros.detectMultiScale(imagen_gris, scaleFactor=1.3, minNeighbors=5)
    return rostros
