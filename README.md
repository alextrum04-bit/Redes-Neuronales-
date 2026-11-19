# Redes Neuronales

Tareas de la materia de Redes Neuronales (Licenciatura en Física, FCFM – BUAP).
Cada carpeta corresponde a una tarea, con su código y las gráficas de resultados.

| # | Tarea | Contenido |
|---|-------|-----------|
| 01 | [Red neuronal en MNIST](01_red_neuronal_mnist/) | Entrenamiento de la red `[784, 30, 10]` del libro de Nielsen con SGD |
| 02 | [Cross-Entropy](02_cross_entropy/) | Se agrega la función de costo de entropía cruzada a `network.py` |
| 03 | [Optimizador Adam](03_adam/) | Implementación de Adam (momentos de primer y segundo orden con corrección de sesgo) |
| 04 | [Red densa secuencial con Keras](04_red_densa_keras/) | Experimentos con activaciones (sigmoid, softmax, tanh), capas ocultas e hiperparámetros |
| 05 | [Regularización](05_regularizacion/) | L1, L2, L1+L2, Dropout y Dropout+L1+L2, con seguimiento de experimentos en MLflow (DagsHub) |
| 06 | [Capa RGB a escala de grises](06_rgb_a_grises/) | Capa personalizada de Keras que convierte imágenes RGB a escala de grises |
| 07 | [Aproximación de funciones](07_aproximacion_funciones/) | Redes que aprenden 3 sin(πx), 1 + 2x + 4x³ y una capa polinomial entrenable que ajusta cos(2x) |
| 08 | [Desbloqueo facial](08_desbloqueo_facial/) | Reconocimiento de mi rostro con transfer learning (MobileNetV2 + CelebA) |

## Estructura

```
├── 01_red_neuronal_mnist/ … 08_desbloqueo_facial/   # una carpeta por tarea
├── libro_nielsen/   # código base del libro "Neural Networks and Deep Learning"
├── data/            # mnist.pkl.gz
└── requirements.txt
```

## Cómo correrlo

```bash
pip install -r requirements.txt
python 01_red_neuronal_mnist/run_network.py
```

Los scripts de las tareas 01–03 importan `mnist_loader` desde `libro_nielsen/` y leen los datos de `data/`, así que se pueden correr desde cualquier carpeta.

## Créditos

El código de `libro_nielsen/` viene del libro [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com) de Michael Nielsen, en la versión para Python 3 de [Michał Dobrzański](https://github.com/MichalDanielDobrzanski/DeepLearningPython) (licencia MIT, ver `libro_nielsen/LICENSE.md`).
