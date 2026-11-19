import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os

IMG_SIZE = 128

MODEL_PATH = r"C:\Users\agust\Documents\Facial Recognition\models\face_classifier.h5"
OUTPUT_MODEL = r"C:\Users\agust\Documents\Facial Recognition\models\final_face_model.h5"

MY_FACE_DIR = r"C:\Users\agust\Documents\Facial Recognition\data\my_face"
OTHERS_DIR = r"C:\Users\agust\Documents\Facial Recognition\data\celeba\img_align_celeba"


# FUNCIÓN DE CARGA DE IMÁGENES
def load_face_data(my_face_dir, others_dir):

    valid_ext = ('.jpg', '.jpeg', '.png')

    my_images = []
    others_images = []

    # --- mis fotos ---
    for fname in os.listdir(my_face_dir):
        if fname.lower().endswith(valid_ext):
            img = cv2.imread(os.path.join(my_face_dir, fname))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            my_images.append(img.astype(np.float32) / 255.0)

    # --- negativos ---
    limit = len(my_images) * 5
    all_others = [f for f in os.listdir(others_dir) if f.lower().endswith(valid_ext)]
    np.random.shuffle(all_others)
    selected = all_others[:limit]

    for fname in selected:
        img = cv2.imread(os.path.join(others_dir, fname))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        others_images.append(img.astype(np.float32) / 255.0)

    X = np.array(my_images + others_images)
    y_labels = np.array([1]*len(my_images) + [0]*len(others_images))
    y_onehot = keras.utils.to_categorical(y_labels, 2)

    return X, y_onehot, y_labels


# FINE TUNING
if __name__ == "__main__":

    model = keras.models.load_model(MODEL_PATH)

    X, y, y_labels = load_face_data(MY_FACE_DIR, OTHERS_DIR)

    # split
    X_train, X_val, y_train, y_val, labels_train, labels_val = train_test_split(
        X, y, y_labels,
        test_size=0.2,
        shuffle=True,
        stratify=y_labels
    )

    # 1) Encontrar última capa convolucional REAL
    last_conv = None
    for i, layer in enumerate(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            last_conv = i

    if last_conv is None:
        raise RuntimeError("ERROR: No se encontraron capas convolucionales.")

    # descongelar últimas capas a partir de la última conv
    unfreeze_from = max(0, last_conv - 3)

    print("\nDescongelando capas desde:", unfreeze_from)

    for layer in model.layers:
        layer.trainable = False

    for layer in model.layers[unfreeze_from:]:
        # evitar descongelar BatchNorm
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

        print("  ", layer.name, "- trainable:", layer.trainable)

    # 2) Compilar con LR pequeño
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 3) Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-7),
        keras.callbacks.ModelCheckpoint(OUTPUT_MODEL, monitor="val_loss", save_best_only=True)
    ]

    # 4) Entrenar
    print("\nIniciando Fine-Tuning...\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=16,
        callbacks=callbacks
    )

    print("\nModelo final guardado en:", OUTPUT_MODEL)
