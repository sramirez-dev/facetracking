# models/cnn_model.py

from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2

def build_cnn_model():
    model = models.Sequential()
    
    # Primera capa convolucional con regularización L2 y Batch Normalization
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1), kernel_regularizer=l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))  # Dropout para reducir el sobreajuste
    
    # Segunda capa convolucional
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    # Tercera capa convolucional
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))  # Dropout más agresivo para prevenir el sobreajuste

    # Flatten y capas densas
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout final para regularización
    model.add(layers.Dense(7, activation='softmax'))  # 7 emociones con softmax
    
    # Compilar el modelo con una tasa de aprendizaje más baja para aprendizaje refinado
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
