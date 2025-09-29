


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from matplotlib import pyplot as plt
import numpy as np
import math

class PolinomioGrado3(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PolinomioGrado3, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.a0 = self.add_weight(name="a0", shape=(1,), initializer='zeros', trainable=True)
        self.a1 = self.add_weight(name="a1", shape=(1,), initializer='zeros', trainable=True)
        self.a2 = self.add_weight(name="a2", shape=(1,), initializer='zeros', trainable=True)
        self.a3 = self.add_weight(name="a3", shape=(1,), initializer='zeros', trainable=True)
        super(PolinomioGrado3, self).build(input_shape)
    
    def call(self, inputs):
        x = inputs
        return self.a0 + self.a1*x + self.a2*tf.pow(x, 2) + self.a3*tf.pow(x, 3)


modelo = keras.Sequential([
    PolinomioGrado3(input_shape=(1,))
])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='mse'
)
_entrenamiento = np.random.uniform(-1, 1, (10000, 1))
y_entrenamiento = np.cos(2 * x_entrenamiento)


# Entrenamiento normal
historia = modelo.fit(
    x_entrenamiento,
    y_entrenamiento,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

capa_polinomio = modelo.layers[0]

print(f"a0 = {capa_polinomio.a0.numpy()[0]:.6f}")
print(f"a1 = {capa_polinomio.a1.numpy()[0]:.6f}")
print(f"a2 = {capa_polinomio.a2.numpy()[0]:.6f}")
print(f"a3 = {capa_polinomio.a3.numpy()[0]:.6f}")

# Gráficos
x_test = np.linspace(-1, 1, 200).reshape(-1, 1)
y_real = np.cos(2 * x_test)
y_pred = modelo.predict(x_test)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(x_test, y_real, 'b-', label='cos(2x) Real', linewidth=2)
plt.plot(x_test, y_pred, 'r--', label='Polinomio Aproximado', linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Aproximación polinómica de grado 3 a cos(2x)')
plt.legend()
plt.grid(True)

plt.show()
