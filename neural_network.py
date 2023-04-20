import random
from numpy import tanh
#Snake neuron.
#For input layer nodes, use the sensor field to plug data dorectly into the neuron.
class Node:
    activationFunction = lambda x : tanh(x) # Tangens hiperboliczny jako funckjca aktywacyjna
    
    def CalcActivationValue(self) -> None:
        if(self.sensor != None):
            self.activationValue = self.sensor
            return self.activationValue
        weighted_average = 0
        weight_sum = 0
        for [node,weight] in inputNodes:
            weight_sum+=weight
            weighted_average += node.activationValue * weight
        weighted_average /= weight_sum
        self.activationValue = Node.activationFunction(weighted_average)
        
    def __init__(self) -> None:
        self.sensor = None # In the first layer, sensor gives the value for the activation function. 
        self.inputNodes = [] # = [Node, weight], [] for sensor nodes
        self.activationValue = None # Cache of the value of the activation function
        
        
    def addInputNode(self,node,initWeight) -> None:
        self.inputNodes.append([node,initWeight])
 
# A genome for our snake.
# Every genome is a full list of weights. 
# Organized as 3 lists of lists
# HiddenLayerOne is all weights of all neurons in Hidden Layer One, etc
class DNA:
    InputLayerSize = 32
    HiddenLayerOneSize = 20
    HiddenLayerTwoSize = 12
    OutputLayerSize = 4
    #Such is life without constuctor overloading...    
    def __init__(self,
                 HL1 = [[random.random()*2.0-1.0 for __ in range(32)] for _ in range(20)], # Python doesnt like if i put the aliases there
                 HL2 = [[random.random()*2.0-1.0 for __ in range(20)] for _ in range(12)], # So more crimes against Python is in order.
                 OL = [[random.random()*2.0-1.0 for __ in range(12)] for _ in range(4)]):
        self.HiddenLayerOne = HL1
        self.HiddenLayerTwo = HL2
        self.OutputLayer = OL
#
# The neural network that controlls the movement of the snake. (its brain)
#
class SnakeBrain:
    
    
    def __init__(self, Chromosome : DNA) -> None:
        #Node init
        self.InputLayer = [Node() for i in range(DNA.InputLayerSize)]
        self.HiddenLayerOne = [Node() for i in range(DNA.HiddenLayerOneSize)]
        self.HiddenLayerTwo = [Node() for i in range(DNA.HiddenLayerTwoSize)]
        self.Output = [Node() for i in range(DNA.OutputLayerSize)] #Use GetOutput to get only one value.
        
        #Applying DNA
        #Hidden layer One
        for hlo in range(DNA.HiddenLayerOneSize):
            for inp in range(DNA.InputLayerSize):
                self.HiddenLayerOne[hlo].addInputNode(self.InputLayer[inp],Chromosome.HiddenLayerOne[hlo][inp])
        #Hidden layer Two
        for hlt in range(DNA.HiddenLayerTwoSize):
            for hlo in range(DNA.HiddenLayerOneSize):
                self.HiddenLayerTwo[hlt].addInputNode(self.HiddenLayerOne[hlo],Chromosome.HiddenLayerTwo[hlt][hlo])
        #Output Layer
        for out in range(DNA.OutputLayerSize):
            for hlt in range(DNA.HiddenLayerTwoSize):
                self.Output[out].addInputNode(self.HiddenLayerTwo[hlt], Chromosome.OutputLayer[out][hlt])       
        
    
    #Sets the sensor values. values List has to be DNA.InputLayerSize long.
    def SetInputValues(self,values: list[int]) -> None:
        for i in range(DNA.InputLayerSize):
            self.InputLayer[i].sensor = values[i]
    
    #Returns the index of the best activated output layer node. Do with that what you will. 
    # In the YT video, that would be 0 == U, 1 == D, 2 == L, 3 == R.
    def GetOutput(self) -> int:
        self.calcOutput()
        maxi = 0
        for x in self.Output:
            if x.activationValue > self.Output[maxi].activationValue:
                maxi = x
        return maxi
    
    # In order to avoid duplicative calculation, we calcuate from bottom up. 
    # You will likely never need to use this function direclty, just use getOutput()
    def calcOutput(self) -> None:
        for i in self.HiddenLayerOne: i.CalcActivationValue()
        for i in self.HiddenLayerTwo: i.CalcActivationValue()
        for i in self.Output: i.CalcActivationValue()



