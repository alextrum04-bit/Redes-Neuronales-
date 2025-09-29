import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer

class RGBToGrayscaleTF(Layer):
    def __init__(self, **kwargs): #Aquí se definen parámetros que la capa necesitará al ser creada
        super().__init__(**kwargs)  #Llama al constructor de Layer con esos argumentos para que la capa los herede.
    
    def call(self, inputs):          #Define qué cálculos se hacen cuando se pasan datos por ella. ´Inputs´ es la emtrada con 3 canales (RGB)




       return tf.image.rgb_to_grayscale(inputs)   #la convierte a 1 canal (escala de grises)
    
    def compute_output_shape(self, input_shape):   #Describe el tamaño de salida de la imagen después de pasar por la cap
        return (input_shape[0], input_shape[1], input_shape[2], 1)