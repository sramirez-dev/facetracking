import signal
import sys
import cv2
from flask import Flask, render_template, Response, jsonify, send_from_directory
import time
import os
from datetime import datetime
import csv
import pytz
from deepface import DeepFace


app = Flask(__name__)

# Variables para el registro de emociones y tiempo
camera = cv2.VideoCapture(0)  # Mover la cámara a nivel global para poder liberarla correctamente
emotion_log = []
emotion_totals = {
    'happy': 0,
    'sad': 0,
    'angry': 0,
    'surprise': 0,
    'neutral': 0,
    'disgust': 0,
    'fear': 0
}
emotion_count = 0  # Contador de cuántos frames con emociones se han registrado
start_time = time.time()
log_interval = 2 * 60  # 2 minutos en segundos
log_folder = "Logs"

# Asegurarse de que la carpeta "Logs" existe
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

# Definir la zona horaria para UTC-6 (América Central)
timezone = pytz.timezone('America/El_Salvador')

# Función para guardar emociones en un archivo CSV
def save_emotion_log(emotions):
    log_file = os.path.join(log_folder, f"emociones_{datetime.now(timezone).strftime('%Y%m%d_%H%M%S')}.csv")
    
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Emotion'])
        
        for emotion in emotions:
            writer.writerow([emotion['time'], emotion['emotion']])

# Función para procesar los frames y detectar emociones
def generar_frames():
    global start_time, emotion_log, emotion_totals, emotion_count, camera

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            # Analizar emociones con DeepFace
            resultado = DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False)
            emociones = resultado[0]['emotion']
            
            # Calcular la suma total de emociones en este frame
            total_emotions = sum(emociones.values())

            # Evitar divisiones por cero
            if total_emotions > 0:
                # Normalizar las emociones para que sumen 100%
                for emotion in emotion_totals:
                    emotion_totals[emotion] += (emociones[emotion] / total_emotions) * 100

                # Actualizar el contador de frames analizados
                emotion_count += 1

            # Registrar emoción dominante
            emocion_dominante = max(emociones, key=emociones.get)
            emotion_log.append({'time': datetime.now(timezone).strftime('%Y-%m-%d %H:%M:%S'), 'emotion': emocion_dominante})

            # Dibujar un rectángulo alrededor del rostro y mostrar la emoción dominante
            if 'region' in resultado[0]:
                x, y, w, h = resultado[0]['region']['x'], resultado[0]['region']['y'], resultado[0]['region']['w'], resultado[0]['region']['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emocion_dominante, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error al detectar emociones: {e}")

        # Guardar el log cada 2 minutos
        current_time = time.time()
        if current_time - start_time >= log_interval:
            save_emotion_log(emotion_log)
            emotion_log = []
            start_time = current_time

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Ruta para enviar los porcentajes de emociones acumuladas
@app.route('/get_emotions')
def get_emotions():
    global emotion_totals, emotion_count
    emotion_percentages = {}

    if emotion_count > 0:
        # Calcular el porcentaje basado en la media acumulada
        for emotion, total in emotion_totals.items():
            emotion_percentages[emotion] = total / emotion_count
    else:
        # Si no se han registrado emociones, los porcentajes son 0
        emotion_percentages = {emotion: 0 for emotion in emotion_totals}

    return jsonify({"emotions": emotion_percentages})

# Función para manejar la señal de interrupción y liberar la cámara
def signal_handler(sig, frame):
    print("Ctrl+C detectado. Cerrando la cámara...")
    camera.release()  # Liberar la cámara
    sys.exit(0)

# Capturar la señal de Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

@app.route('/')
def index():
    log_files = os.listdir(log_folder)
    return render_template('index.html', log_files=log_files)

@app.route('/video_feed')
def video_feed():
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download_log/<filename>')
def download_log(filename):
    return send_from_directory(log_folder, filename)

if __name__ == "__main__":
    app.run(debug=True)
