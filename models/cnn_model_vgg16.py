from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

def build_vgg16_model():
    # Cargamos VGG16 preentrenado y descongelamos las últimas 4 capas
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

    for layer in vgg.layers[:-4]:  # Congelamos todas las capas excepto las últimas 4
        layer.trainable = False

    model = models.Sequential()

    # Añadimos VGG16 como extractor de características
    model.add(vgg)
    
    # Capas densas para la clasificación de emociones
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # Regularización
    model.add(layers.Dense(7, activation='softmax'))  # 7 emociones

    # Compilamos el modelo con un optimizador más refinado
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
