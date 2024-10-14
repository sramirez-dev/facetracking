# utils/preprocessing.py

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data(csv_file):
    data = pd.read_csv(csv_file)
    pixels = data['pixels'].tolist()
    images = np.array([np.fromstring(pix, dtype=int, sep=' ').reshape(48, 48, 1) for pix in pixels])
    labels = pd.get_dummies(data['emotion']).values
    return images, labels

def create_data_generator(images, labels):
    # Aumento de datos con rotaciones, zoom y otros cambios
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    
    # Generador de datos con aumento
    data_gen = datagen.flow(images, labels, batch_size=64)
    return data_gen
