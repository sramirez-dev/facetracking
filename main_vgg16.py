from models.cnn_model_vgg16 import build_vgg16_model
from utils.preprocessing_vgg import load_and_preprocess_data_vgg, create_data_generator_vgg
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

# Diccionario de emociones
emotion_dict = {0: "Enojo", 1: "Disgusto", 2: "Miedo", 3: "Felicidad", 4: "Tristeza", 5: "Sorpresa", 6: "Neutral"}

# Nombre del archivo donde se guardará el modelo
model_file = "emotion_recognition_vgg16.h5"

# Cargar los datos y preprocesarlos
images, labels = load_and_preprocess_data_vgg('fer2013.csv')

# Crear el generador de datos con aumento
data_gen = create_data_generator_vgg(images, labels)

# Verificar si el modelo ya ha sido entrenado y guardado
if os.path.exists(model_file):
    print("Cargando el modelo entrenado...")
    model = load_model(model_file)
else:
    print("Entrenando el modelo por primera vez...")
    # Construir el modelo VGG16
    model = build_vgg16_model()

    # Entrenar el modelo
    model.fit(data_gen, epochs=30)

    # Guardar el modelo entrenado para uso futuro
    model.save(model_file)
    print(f"Modelo guardado como {model_file}")

# Detección de emociones en tiempo real
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_emotions():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (48, 48)).reshape(1, 48, 48, 3)
            emotion_prediction = model.predict(face_resized)
            emotion_label = np.argmax(emotion_prediction)
            emotion_text = emotion_dict[emotion_label]

            # Dibujar el rectángulo y mostrar la emoción detectada
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detect_emotions()
