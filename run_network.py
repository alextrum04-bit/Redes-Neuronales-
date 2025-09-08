import mnist_loader #Abre el archivo que ya esta en el repositorio de DeepLearning el cual tiene funciones para cargar el dataset
from network import Network, CrossEntropyCost #Se importa la clase Network y la funcion de costo CrossEntropyCost desde el archivo network.py


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()    #Aqui se leen los datos que hay en el archivo mnist_loader

#Se devuelven tres conjuntos de datos. El primero es para entrenar la red, el segundo es para validar la red durante el entrenamiento
#y el tercero para probar la red final.



net = Network([784, 30, 10], cost=CrossEntropyCost)
#Crea una red neuronal nueva a partir de la clase Network que está definida en network.py.
#cost=CrossEntropyCost indica que se usara la entropía cruzada (en lugar de MSE)



net.SGD(training_data, 15, 7, 0.8, test_data=test_data)    #Se crea la instruccion que entrena la red neuronal con el metodo SGD