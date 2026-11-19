import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np

class Funsol(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        batch_size = 100  # Aumentar batch size para mejor estabilidad
        x = tf.random.uniform((batch_size, 1), minval=-1, maxval=1)  # Shape (batch_size, 1)

        # Función objetivo: 1 + 2x + 4x³
        eq = 1 + 2*x + 4*tf.pow(x, 3)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # Usar MSE directamente
            loss = tf.reduce_mean(tf.square(y_pred - eq))

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

# Construir modelo
inputs = keras.Input(shape=(1,), name='input_layer') # Changed input shape to (1,)
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(1, activation='linear')(x) # Changed output shape to (1,) and activation to linear

model = Funsol(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(), loss='mse')


# Entrenar por más épocas
history = model.fit(x=tf.zeros((1,1)), y=tf.zeros((1,1)), epochs=500, verbose=1)

# Evaluar
x_test = np.linspace(-1, 1, 100).reshape(-1, 1)
y_true = 1 + 2*x_test + 4*(x_test**3)
y_pred = model.predict(x_test)

# Graficar
plt.figure(figsize=(10,6))
plt.plot(x_test, y_true, label="Función real: 1+2x+4x³", linewidth=2, color="black")
plt.plot(x_test, y_pred, "--", label="Predicción red", linewidth=2, color="red")
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Aproximación de 1 + 2x + 4x³ con Funsol (Corregido)")
plt.grid(True, alpha=0.3)
plt.show()
