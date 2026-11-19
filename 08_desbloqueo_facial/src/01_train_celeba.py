import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import os
import time
import traceback

# Configuración
IMG_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 10

def load_celeba(data_dir, attr_file, num_samples=None):
    attr_path = os.path.join(data_dir, attr_file)
    
    if not os.path.exists(attr_path):
        raise FileNotFoundError(f"No se encuentra: {attr_path}")

    # Leer SIN usar ninguna columna como índice
    df_attr = pd.read_csv(attr_path, sep=r'\s+', skiprows=1, index_col=False)

    img_dir = os.path.join(data_dir, 'img_align_celeba')
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"No se encuentra: {img_dir}")

    if num_samples and num_samples < len(df_attr):
        df_attr = df_attr.iloc[:num_samples]

    # Primera columna = nombres de archivo
    filenames = df_attr.iloc[:, 0].astype(str).values
    image_paths = [os.path.join(img_dir, f) for f in filenames]

    # Lista de atributos
    attributes = ['Male', 'Young', 'Eyeglasses', 'Smiling', 'Wearing_Hat']

    for attr in attributes:
        if attr not in df_attr.columns:
            raise ValueError(f"Atributo {attr} no encontrado")

    labels = df_attr[attributes].values
    labels = (labels + 1) // 2

    for i, attr in enumerate(attributes):
        percent = np.mean(labels[:, i]) * 100
        print(f"  {attr}: {percent:.1f}%")

    return image_paths, labels


@tf.function
def preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def create_optimized_dataset(image_paths, labels, batch_size, shuffle_buffer=1000):
    num_samples = len(image_paths)
    
    # Emparejar a múltiplos del batch
    num_samples = (num_samples // batch_size) * batch_size
    image_paths = image_paths[:num_samples]
    labels = labels[:num_samples]
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(
        preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, num_samples

def create_attr_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_attrs=5):
    base = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',   # usa pesos pre-entrenados
        pooling='avg'
    )
    
    # Congelar primeras capas
    for layer in base.layers[:100]:
        layer.trainable = False
    
    x = base.output
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_attrs, activation='sigmoid')(x)
    
    model = keras.Model(inputs=base.input, outputs=outputs)
    return model

def main():
    try:
        # Detectar GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            print("No se detectó GPU, usando CPU")
        
        # Rutas
        data_dir = r"C:\Users\agust\Documents\Facial Recognition\data\celeba"
        models_dir = r"C:\Users\agust\Documents\Facial Recognition\models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Cargar datos (pocos para prueba rápida)
        image_paths, labels = load_celeba(
            data_dir,
            "list_attr_celeba.txt",
            num_samples=2000
        )
        
        dataset, num_samples = create_optimized_dataset(
            image_paths, labels, BATCH_SIZE
        )
        
        # Dividir dataset
        total_batches = num_samples // BATCH_SIZE
        train_batches = int(0.8 * total_batches)
        val_batches = total_batches - train_batches
        
        train_dataset = dataset.take(train_batches)
        val_dataset = dataset.skip(train_batches)
        
        # Crear modelo
        model = create_attr_model()
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc', multi_label=True)
            ]
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(models_dir, "best_model.keras"),
                monitor='val_auc',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Entrenar
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            steps_per_epoch=train_batches,
            validation_steps=val_batches,
            callbacks=callbacks,
            verbose=1
        )
        
        # Guardar modelo final
        save_path = os.path.join(models_dir, "pretrained_celeba.keras")
        model.save(save_path)
        print(f"\n Modelo guardado en: {save_path}")
        
        return True
    
    except Exception as e:
        print(f"\n ERROR durante la ejecución:")
        print(f"Mensaje: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
