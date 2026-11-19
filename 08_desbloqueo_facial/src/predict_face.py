import cv2
import os
import random
import numpy as np
import tensorflow as tf


class FaceRecognizer:
    def __init__(self, model_path, threshold=0.8):
        # Cargar modelo final
        self.model = tf.keras.models.load_model(model_path)
        self.threshold = threshold

        # Detector Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    # Detectar rostro en una imagen completa
    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4
        )
        return faces

    # Preprocesamiento igual al entrenamiento
    def preprocess_face(self, roi):
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)     # BGR → RGB
        roi = cv2.resize(roi, (128, 128))              # tamaño del modelo
        roi = roi.astype(np.float32) / 255.0           # normalización
        return np.expand_dims(roi, axis=0)             # (1,128,128,3)

    # Clasificar una imagen completa
    def classify_image(self, image):
        faces = self.detect_face(image)

        if len(faces) == 0:
            return image, "SIN CARA DETECTADA"

        # Tomar la primera cara detectada
        (x, y, w, h) = faces[0]
        face_roi = image[y:y+h, x:x+w]

        # Preprocesar igual que entrenamiento
        processed = self.preprocess_face(face_roi)

        # Predicción del modelo
        pred = self.model.predict(processed, verbose=0)[0]
        prob_me = pred[1]      # clase “YO”

        # Asignación según umbral
        if prob_me > self.threshold:
            label = f"TÚ ({prob_me:.2f})"
            color = (0, 255, 0)
        else:
            label = f"OTRO ({prob_me:.2f})"
            color = (0, 0, 255)

        # Dibujar el recuadro y etiqueta
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            image, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

        return image, label


# Seleccionar varias imágenes aleatorias de una carpeta
def load_random_images(folder, count=5):
    valid = (".jpg", ".jpeg", ".png")
    files = [os.path.join(folder, f)
             for f in os.listdir(folder)
             if f.lower().endswith(valid)]

    if len(files) < count:
        count = len(files)

    return random.sample(files, count)


# Crear un grid visual (2x5)
def create_grid(images, rows=2, cols=5, img_size=(300, 300)):
    resized = [cv2.resize(img, img_size) for img in images]
    grid = np.vstack([
        np.hstack(resized[i*cols:(i+1)*cols]) for i in range(rows)
    ])
    return grid


# MAIN – PRUEBA DEL MODELO
if __name__ == "__main__":
    recognizer = FaceRecognizer(
        r"C:/Users/agust/Documents/Facial Recognition/models/final_face_model.h5"
    )

    # 5 fotos tuyas
    my_photos = load_random_images(
        r"C:/Users/agust/Documents/Facial Recognition/data/my_face", 5
    )

    # 5 fotos de CelebA
    celeba_photos = load_random_images(
        r"C:/Users/agust/Documents/Facial Recognition/data/celeba/img_align_celeba", 5
    )

    all_processed = []

    # Clasificar cada imagen
    for path in my_photos + celeba_photos:
        img = cv2.imread(path)
        if img is None:
            print("Error leyendo:", path)
            continue

        processed, label = recognizer.classify_image(img)
        all_processed.append(processed)

    # Crear grid 2x5
    grid = create_grid(all_processed, rows=2, cols=5)

    cv2.imshow("5 Fotos Mías + 5 de CelebA (Clasificadas)", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
