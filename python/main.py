import numpy as np
import pandas as pd
import os
import tensorflow as tf
import cv2
import random
from tensorflow import keras
from tensorflow.keras import layers, Input
from keras.layers.core import Dense
from tensorflow.keras.models import Sequential, Model
#Folder en donde las imagenes estan etiquetadas
folder = 'images/'

#Para que todas las imagenes tengan el mismo tamaño
IMG_WIDTH = 600
IMG_HEIGHT = 600

def create_dataset(folder):

  image_array = []
  label_array = []

  #Atravez de todas las etiquetas (aka botella y no botella)
  for label in os.listdir(folder):
    #Sobre todos los archivos (aka las fotos)
    for file in os.listdir(os.path.join(folder, label)):
      
      #Carga la imagen
      image_path=os.path.join(folder, label, file)
      image=cv2.imread( image_path, cv2.COLOR_BGR2RGB)

      #Cambiarle tamaño a la imagen
      image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)

      #Convertirla en array numpy (matriz de RED GREEN BLUE RGB)
      image=np.array(image)

      #Convertir el array a un numero flotante
      image=image.astype('float32')
      
      #Normalizar
      image /= 255

      image_array.append(image)
      label_array.append(label)
  
  #Random shuffle
  for r in range(len(image_array)):
    copy1 = image_array[r]
    copy2 = label_array[r]
    pos = random.randint(0, len(image_array) -1)
    image_array[r] = image_array[pos]
    label_array[r] = label_array[pos]
    image_array[pos] = copy1
    label_array[pos] = copy2
  return image_array, label_array

img_data, label_data = create_dataset(folder)

#Crea un diccionario de los nombres (botella, no_botella) a numeros (0,1)
target_dict = {k: v for v, k in enumerate(np.unique(label_data))}

#label_data (string) -> target_val (numero)
target_val = [target_dict[label_data[i]] for i in range(len(label_data))]

#Prepara el modelo
model=tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(IMG_HEIGHT,IMG_WIDTH, 3)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(6)
        ])

model.compile(optimizer='Adadelta', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  
#Prueba a encajar el modelo
history = model.fit(x=np.array(img_data, np.float32), y=np.array(list(map(int,target_val)), np.float32), epochs=50)

print(history)














