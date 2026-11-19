


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from matplotlib import pyplot as plt
import numpy as np
import math

loss_tracker = keras.metrics.Mean(name="loss")

class Funsol(keras.Model):
    @property
    def metrics(self):
        return [loss_tracker] #igual cambia el loss_tracker

    def train_step(self, data):
        batch_size = 20 #Calibra la resolucion de la ec.dif
        x = tf.random.uniform((batch_size,), minval=-1, maxval=1)
        eq = 1 + 2*x + 4*tf.pow(x, 3)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = tf.keras.losses.MeanSquaredError()(y_pred,eq)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        #actualiza metricas
        loss_tracker.update_state(loss)

        return {"loss": loss_tracker.result()}



inputs = keras.Input(shape=(1,))
x = keras.layers.Dense(64, activation="tanh")(inputs)
x = keras.layers.Dense(64, activation="tanh")(x)
outputs = keras.layers.Dense(1)(x)

model = Funsol(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(0.01))
model.summary()


history = model.fit(x=tf.zeros((1,1)), y=tf.zeros((1,1)), epochs=100, verbose=1)

x_test = np.linspace(-1, 1, 100).reshape(-1, 1)
y_true = 1 + 2*x_test + 4*(x_test**3)
y_pred = model.predict(x_test)

plt.figure(figsize=(8,5))
plt.plot(x_test, y_true, label="Función real: 1+2x+4x³", color="black")
plt.plot(x_test, y_pred, "--", label="Predicción red", color="red")
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Aproximación de 1 + 2x + 4x³ con Funsol")
plt.show()
