import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data_vgg(csv_file):
    data = pd.read_csv(csv_file)
    pixels = data['pixels'].tolist()
    images = np.array([np.fromstring(pix, dtype=int, sep=' ').reshape(48, 48, 1) for pix in pixels])
    # Convertimos las imágenes de escala de grises a RGB
    images = np.array([np.concatenate([image] * 3, axis=-1) for image in images])
    labels = pd.get_dummies(data['emotion']).values
    return images, labels

def create_data_generator_vgg(images, labels):
    datagen = ImageDataGenerator(
        rotation_range=20,       # Aumento más agresivo
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    
    data_gen = datagen.flow(images, labels, batch_size=64)
    return data_gen
