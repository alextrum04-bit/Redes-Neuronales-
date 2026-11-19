# %load network.py

"""
network.py
~~~~~~~~~~
IT WORKS

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

#Cross-Entropy

class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        """Devuelve el costo para la salida `a` y el resultado deseado "y"."""
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Devuelve el error delta en la última capa."""
        # Con cross-entropy se simplifica: no depende de la derivada de la sigmoide
        return (a - y)
#---------------------

class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        #Adam necesita momentos de primer y segundo orden (m y v) para cada peso y sesgo.
        self.m_w = [np.zeros(w.shape) for w in self.weights]
        self.v_w = [np.zeros(w.shape) for w in self.weights]
        self.m_b = [np.zeros(b.shape) for b in self.biases]
        self.v_b = [np.zeros(b.shape) for b in self.biases]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
        test_data=None, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        t = 0  # contador global de mini-batches
         
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                t += 1
                self.update_mini_batch_adam(mini_batch, eta, t, beta1, beta2, epsilon)
            if test_data:
                print("Epoch {} : {} / {}".format(j+1, self.evaluate(test_data), n_test))
            
            else:
                print("Epoch {} complete".format(j+1))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def update_mini_batch_adam(self, mini_batch, eta, t, beta1, beta2, epsilon):
        #Actualiza pesos y biases usando Adam en un mini-batch
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
    
    
            # Acumulamos gradientes con backprop
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # Promedio del mini-batch
        nabla_b = [nb / len(mini_batch) for nb in nabla_b]
        nabla_w = [nw / len(mini_batch) for nw in nabla_w]
         
         # Actualización Adam
        self.m_w = [beta1 * mw + (1-beta1) * nw for mw, nw in zip(self.m_w, nabla_w)]
        self.v_w = [beta2 * vw + (1-beta2) * (nw**2) for vw, nw in zip(self.v_w, nabla_w)]
        self.m_b = [beta1 * mb + (1-beta1) * nb for mb, nb in zip(self.m_b, nabla_b)]
        self.v_b = [beta2 * vb + (1-beta2) * (nb**2) for vb, nb in zip(self.v_b, nabla_b)]
         
         # Corrección de sesgo
        m_w_hat = [mw / (1-beta1**t) for mw in self.m_w]
        v_w_hat = [vw / (1-beta2**t) for vw in self.v_w]
        m_b_hat = [mb / (1-beta1**t) for mb in self.m_b]
        v_b_hat = [vb / (1-beta2**t) for vb in self.v_b]
         
         # Actualización final
        self.weights = [w - eta * mw / (np.sqrt(vw) + epsilon)
                    for w, mw, vw in zip(self.weights, m_w_hat, v_w_hat)]
        
        self.biases = [b - eta * mb / (np.sqrt(vb) + epsilon)
                   for b, mb, vb in zip(self.biases, m_b_hat, v_b_hat)]

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


