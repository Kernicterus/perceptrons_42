import numpy as np
import pandas as pd

def sigmoid(z):
    """
    Function that calculates the sigmoid of a given value
    alias g(z) and returns it
    """    
    return 1 / (1 + np.exp(-z))


def relu(z) :
    """
    Function that calculates the RELU of a given vector z
    and returns it
    """   
    return np.maximum(0, z)


def softmax(z) :
    """
    Function that calculates the softmax of a given vector z
    and returns it
    """    
    zMax = np.max(z)
    zStable = z - zMax
    exponentiation = np.exp(zStable)
    sumExp = np.sum(exponentiation)
    return exponentiation / sumExp


def binaryCrossEntropy() :
    pass
    

functionMap = {
    "sigmoid" : sigmoid,
    "relu" : relu,
    "softmax" : softmax
}

lossMap = {
    "binaryCrossEntropy" : binaryCrossEntropy
}


def forwardPropagation(newWeights : list[np.ndarray], biases : list, batchData, activationsByLayer : list) -> tuple:
    cacheZ = {}
    cacheZb = {}
    cacheA = {}
    for i in range(len(newWeights)) :
        idZ = f"l{i}"
        if i == 0 :
            inputValues = batchData
        else :
            inputValues = cacheA[f"l{i - 1}"]
        # print(i)
        # print(activationsByLayer[i])
        # print(f"layer {i} :shape weigths {newWeights[i].shape} - shape input {inputValues.shape}")
        cacheZ[idZ] = np.dot(newWeights[i], inputValues)
        # print (f"cache Z l{i} : {cacheZ[idZ]}")
        cacheZb[idZ] = cacheZ[idZ] + biases[i]
        # print (f"cache Z l{i} + bias : {cacheZb[idZ]}")
        cacheA[f"l{i}"] = functionMap[activationsByLayer[i]](cacheZb[idZ])
        # print (f"cache A {f"l{i}"}: {cacheA[f"l{i}"]}")

    return cacheA, cacheZb


def backwardPropagation(yRealResults : np.ndarray, caches : tuple, newWeights : list[np.ndarray], activationByLayer : list) -> list[np.ndarray]:

    return newWeights