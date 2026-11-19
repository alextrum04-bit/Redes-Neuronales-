import mnist_loader #Abre el archivo que ya esta en el repositorio de DeepLearning el cual tiene funciones para cargar el dataset
import network      #Abre el archivo Network el cual implementa una red neuronal con backpropagation y entrenamiento por SGD


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()    #Aqui se leen los datos que hay en el archivo mnist_loader

#Se devuelven tres conjuntos de datos. El primero es para entrenar la red, el segundo es para validar la red durante el entrenamiento
#y el tercero para probar la red final.



net = network.Network([784, 30, 10])      #Crea una red neuronal nueva a partir de la clase Network que está definida en network.py.

net.SGD(training_data, 15, 7, 0.8, test_data=test_data)    #Se crea la instruccion que entrena la red neuronal con el metodo SGD
