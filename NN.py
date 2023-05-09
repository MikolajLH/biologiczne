from typing import Any
import numpy as np
from copy import deepcopy



def softmax(x : np.array):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def relu(x : np.array, p = 1.):
    return p * x * (x > 0)


class NeuralNetwork:
    def __init__(self) -> None:
        self.__shape = []

        self.__input_layer_added  = False
        self.__output_layer_added = False

        self.__layers : list[tuple[np.array, np.ufunc]] = []
        self.weights : list[np.array] = []


    def copy(self):
        return deepcopy(self)
    

    def add_input_layer(self, input_layer_size : int):
        assert not self.__input_layer_added
        assert not self.__output_layer_added

        self.__shape += [input_layer_size]

        self.__input_layer_added = True


    def add_hidden_layer(self, hidden_layer_size : int, activation_function) -> None:

        assert self.__input_layer_added
        assert not self.__output_layer_added

        self.__shape += [hidden_layer_size]
        self.__layers += [(np.zeros(hidden_layer_size), activation_function)]
        self.weights += [np.ones((self.__shape[-1], self.__shape[-2]))]

    
    def add_output_layer(self, output_layer_size : int, activation_function):

        assert self.__input_layer_added
        assert not self.__output_layer_added

        self.__shape += [output_layer_size]
        self.__layers += [(np.zeros(output_layer_size), activation_function)]
        self.weights += [np.ones((self.__shape[-1], self.__shape[-2]))]

        self.__output_layer_added = True
        

    def __call__(self, input_layer : np.array ) -> Any:
        assert self.__input_layer_added
        assert self.__output_layer_added
        assert len(input_layer) == self.__shape[0]

        Y = np.array(input_layer)
        for W, (l, af) in zip(self.weights, self.__layers):
            Y = af(W @ Y)

        return Y
    

    def W(self, i : int, new_W  : np.array):
        assert np.shape(self.weights[i]) == np.shape(new_W)
        self.weights[i] = new_W


    def u_rand(self, a = -1., b = 1.):
        assert self.__input_layer_added
        assert self.__output_layer_added

        for i, W in  enumerate(self.weights):
            self.weights[i] = np.random.uniform(a,b, np.shape(W))


    def g_rand(self, mean = 0., stddev = 1.):
        assert self.__input_layer_added
        assert self.__output_layer_added

        for i, W in enumerate(self.weights):
            self.weights[i] = np.random.normal(mean, stddev, np.shape(W))


    def show(self):
        print(f"shape: {self.__shape}")
        print("weights:\n")
        for i, w in enumerate(self.weights):
            print(np.shape(w), "==", (self.__shape[i + 1], self.__shape[i]))
            print(w)
            print()
        pass

    def save(self, path : str):
        np.savez(path, *self.weights)

    def load(self, path : str):
        npzfile = np.load(path)
        for i, arr in enumerate(npzfile.files):
            self.W(i, npzfile[arr])
    

def crossover(m : NeuralNetwork, f : NeuralNetwork) -> NeuralNetwork:
    assert len(m.weights) == len(f.weights)

    c = m.copy()
    for i in range(len(c.weights)):
        N,_ = np.shape(c.weights[i])
        for k in range(N):
            if np.random.rand() < 0.5:
                c.weights[i][k] = f.weights[i][k]
    return c



def g_mutate(nn : NeuralNetwork, rate : float, mean = 0., stddev = 1.) -> NeuralNetwork:
    cnn = nn.copy()
    for i in range(len(cnn.weights)):
        R, C = np.shape(cnn.weights[i])
        for r in range(R):
            for c in range(C):
                if np.random.rand() < rate:
                    cnn.weights[i][r,c] += np.random.normal(mean,stddev)
    return cnn


def g_neuron_mutation(nn : NeuralNetwork, rate, K = 2, mean = 0, stddev = 1) -> NeuralNetwork:
    c = nn.copy()

    if np.random.rand() < rate:
        to_change = np.random.choice(len(c.weights), K, False)
        for i in to_change:
            R, C = np.shape(c.weights[i])
            c.weights[i][np.random.randint(0, R)] = np.random.normal(mean, stddev, C)

    return c
    
    z = 0
    for i in range(len(c.weights)):
        N, M = np.shape(c.weights[i])
        for k in range(N):
            c.weights[i][k] = np.random.normal(mean, stddev, M)
            z+=1
            if z == K:
                return c
    return c


if __name__ == "__main__":
    nn = NeuralNetwork()

    nn.add_input_layer(3)
    nn.add_hidden_layer(2, lambda x: x)
    nn.add_output_layer(3, softmax)

    nn.g_rand()

    nn.show()
    

    #nn.W(0, np.zeros((2,3)))
    #nn.W(1, np.zeros((1,2)))


    #nn.save("test")


    #m = nn.copy()
    #f = nn.copy()




