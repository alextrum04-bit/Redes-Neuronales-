import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os

IMG_SIZE = 128

PRETRAINED_MODEL = r"C:\Users\agust\Documents\Facial Recognition\models\pretrained_celeba.keras"
OUTPUT_MODEL = r"C:\Users\agust\Documents\Facial Recognition\models\face_classifier.h5"

MY_FACE_DIR = r"C:\Users\agust\Documents\Facial Recognition\data\my_face"
OTHERS_DIR = r"C:\Users\agust\Documents\Facial Recognition\data\celeba\img_align_celeba"


# CARGA DE MIS FOTOS + FOTOS NEGATIVAS
def load_face_data(my_face_dir, others_dir, img_size=128):

    valid_ext = ('.jpg', '.jpeg', '.png')

    my_images = []
    others_images = []

    # --- Mis fotos ---
    for fname in os.listdir(my_face_dir):
        if fname.lower().endswith(valid_ext):
            path = os.path.join(my_face_dir, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            my_images.append(img.astype(np.float32) / 255.0)

    print("Mis fotos:", len(my_images))

    # --- Negativos (selección aleatoria) ---
    limit = len(my_images) * 5
    all_others = [f for f in os.listdir(others_dir) if f.lower().endswith(valid_ext)]
    np.random.shuffle(all_others)
    selected = all_others[:limit]

    for fname in selected:
        path = os.path.join(others_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        others_images.append(img.astype(np.float32) / 255.0)

    print("Otros rostros:", len(others_images))

    # Etiquetas
    X = np.array(my_images + others_images)
    y_labels = np.array([1]*len(my_images) + [0]*len(others_images))
    y_onehot = keras.utils.to_categorical(y_labels, 2)

    return X, y_onehot, y_labels


# MODELO
def build_face_classifier(pretrained_path):

    pretrained = keras.models.load_model(pretrained_path)

    base = keras.Model(
        inputs=pretrained.input,
        outputs=pretrained.get_layer(index=-3).output
    )

    base.trainable = False   # congelado en esta etapa

    # Clasificador nuevo
    x = layers.Dense(128, activation='relu')(base.output)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(2, activation='softmax')(x)

    model = keras.Model(inputs=base.input, outputs=out)
    return model


# ENTRENAMIENTO
if __name__ == "__main__":

    # cargar dataset
    X, y, y_labels = load_face_data(MY_FACE_DIR, OTHERS_DIR)

    # split estratificado
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y_labels
    )

    # modelo
    model = build_face_classifier(PRETRAINED_MODEL)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # augment
    datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True
    )

    datagen.fit(X_train)

    # entrenar
    model.fit(
        datagen.flow(X_train, y_train, batch_size=16),
        epochs=20,
        validation_data=(X_val, y_val)
    )

    # guardar
    model.save(OUTPUT_MODEL)
    print("\nModelo guardado en:", OUTPUT_MODEL)
