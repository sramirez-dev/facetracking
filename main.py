import cv2
import time
from collections import Counter
from deepface import DeepFace
from utils.face_detection import detect_faces

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Variables para guardar emociones y tiempo
emotion_log = []
start_time = time.time()
log_interval = 2 * 60  # 2 minutos en segundos

# Contador para emociones (excluyendo "neutral")
emotion_counter = Counter()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detección de rostros
    faces = detect_faces(frame)

    # Analizar emociones para cada rostro detectado
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        
        # Predecir emociones con enforce_detection=False para evitar errores si no detecta bien la cara
        result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)

        # Si detecta múltiples rostros, 'result' será una lista
        if isinstance(result, list):
            emotion = result[0]['dominant_emotion']  # Tomar la emoción del primer rostro
        else:
            emotion = result['dominant_emotion']  # Si solo detecta un rostro

        # Excluir la emoción "neutral" del conteo principal
        if emotion != 'neutral':
            emotion_counter[emotion] += 1
        
        # Guardar emoción en el log
        emotion_log.append(emotion)

        # Dibujar rectángulo y mostrar emoción actual en pantalla
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Calcular la emoción dominante (excluyendo "neutral")
    if emotion_counter:
        dominant_emotion = emotion_counter.most_common(1)[0][0]
        cv2.putText(frame, f"Dominant Emotion: {dominant_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Mostrar el frame en la ventana
    cv2.imshow('Emotion Recognition', frame)

    # Guardar logs cada 2 minutos
    if time.time() - start_time > log_interval:
        with open(f'emotion_log_{int(time.time())}.txt', 'w') as f:
            f.write("\n".join(emotion_log))
        emotion_log = []  # Limpiar el log para los próximos 2 minutos
        start_time = time.time()  # Reiniciar el temporizador

    # Opción de detener con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
