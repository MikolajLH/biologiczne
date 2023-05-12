from __future__ import annotations
import numpy as np
from typing import Any, List
from copy import deepcopy


class NeuralNetwork:
    def __init__(self) -> None:
        self.__input_layer_added  = False
        self.__output_layer_added = False

        #temporary
        self.apples = 0
        self.steps = 0

        self.__shape : List[int] = []
        self.weights : List[np.ndarray] = []
        self.biases : List[np.ndarray] = []
        self.activation_functions : List[np.ufunc] = []

    @property
    def shape(self) -> List[int]:
        return self.__shape

    def copy(self) -> NeuralNetwork:
        return deepcopy(self)
    
    def add_input_layer(self, input_layer_size : int) -> None:
        assert not self.__input_layer_added and not self.__output_layer_added

        self.__shape += [input_layer_size]

        self.__input_layer_added = True

    def add_hidden_layer(self, hidden_layer_size : int, activation_function : np.ufunc) -> None:
        assert self.__input_layer_added and not self.__output_layer_added

        self.__shape += [hidden_layer_size]
        self.activation_functions += [activation_function]
        self.biases += [np.zeros(hidden_layer_size)]
        self.weights += [np.ones((self.__shape[-1], self.__shape[-2]))]

    def add_output_layer(self, output_layer_size : int, activation_function) -> None:
        assert self.__input_layer_added and not self.__output_layer_added
        
        self.__shape += [output_layer_size]
        self.activation_functions += [activation_function]
        self.weights += [np.ones((self.__shape[-1], self.__shape[-2]))]
        self.biases += [np.zeros(output_layer_size)]
        
        self.__output_layer_added = True
    
    def __call__(self, input_layer : np.ndarray) -> np.ndarray:
        assert self.__input_layer_added and self.__output_layer_added
        assert len(input_layer) == self.__shape[0]

        y = np.array(input_layer)
        for w, af, b in zip(self.weights, self.activation_functions, self.biases):
            y = af(w @ y - b)

        return y

    def u_rand(self, a= -1, b = 1) -> NeuralNetwork:
        new = self.copy()
        for i, w in enumerate(new.weights):
            new.weights[i] = np.random.uniform(a,b, w.shape)

        for i, b in enumerate(new.biases):
            new.biases[i] = np.random.uniform(a,b, b.shape)
            
        return new

    def g_rand(self, mean : float = 0, stddev : float = 1) -> NeuralNetwork:
        new = self.copy()
        for i, w in enumerate(new.weights):
            new.weights[i] = np.random.normal(mean, stddev, w.shape)

        for i, b in enumerate(new.biases):
            new.biases[i] = np.random.normal(mean, stddev, b.shape)

        return new
    
    def save(self, path : str):
        np.savez(path, *self.weights, *self.biases)

    def load_dep(self, path : str):
        npzfile = np.load(path)
        for i, arr in enumerate(npzfile.files):
            self.weights[i] = npzfile[arr]


    def load(self, path : str):
        npzfile = np.load(path)
        N = len(npzfile.files)

        for i, arr in enumerate(npzfile.files):
            if i < N // 2:
                self.weights[i] = npzfile[arr]
            else:
                self.biases[i - N // 2] = npzfile[arr]
            

if __name__ == "__main__":
    pass