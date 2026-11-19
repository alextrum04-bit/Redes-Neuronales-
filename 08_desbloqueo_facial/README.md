# Desbloqueo facial

Clasificador "mi rostro vs. otros" usando transfer learning en tres etapas.

```
08_desbloqueo_facial/
├── data/
│   ├── celeba/      # CelebA (img_align_celeba + list_attr_celeba.txt), no incluido
│   └── my_face/     # 50 fotos propias, no incluidas
├── models/          # modelos entrenados (.keras / .h5), no incluidos
└── src/
    ├── 01_train_celeba.py
    ├── 02_build_face_classifier.py
    ├── 03_fine_tune.py
    └── predict_face.py
```

| Script | Qué hace |
|--------|----------|
| `01_train_celeba.py` | Entrena MobileNetV2 (pesos de ImageNet, primeras 100 capas congeladas) para predecir 5 atributos de CelebA: Male, Young, Eyeglasses, Smiling, Wearing_Hat. Guarda `pretrained_celeba.keras`. |
| `02_build_face_classifier.py` | Usa la parte convolucional del modelo anterior (congelada) y agrega `Dense(128) → Dropout → Dense(2, softmax)`. Entrena con mis fotos + 5 veces más rostros de CelebA, con data augmentation. Guarda `face_classifier.h5`. |
| `03_fine_tune.py` | Descongela las últimas capas convolucionales (BatchNorm congeladas) y ajusta con lr = 1e-5, EarlyStopping, ReduceLROnPlateau y ModelCheckpoint. Guarda `final_face_model.h5`. |
| `predict_face.py` | Detecta el rostro con Haar Cascade, lo clasifica con umbral 0.8 y muestra un grid de 2×5 con 5 fotos mías y 5 de CelebA. |

Las rutas a los datos y modelos están definidas al inicio de cada script; hay que ajustarlas a la ubicación local.
