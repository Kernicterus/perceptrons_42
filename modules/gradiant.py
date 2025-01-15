import numpy as np
import pandas as pd

def sigmoid(z):
    """
    Function that calculates the sigmoid of a given value
    alias g(z) and returns it
    """    
    return 1 / (1 + np.exp(-z))

def relu(z) :
    pass


def softmax(z) :
    return 1 / (1 + np.exp(-z)) #!!!!!!!!!! to UPD


def binaryCrossEntropy() :
    pass
    

def predictionH0(weights : pd.Series, dfLine : pd.Series):
    """
    Function that calculates h0(x)
    by sending 0Tx to the sigmoid function
    which is the dot product of the weights and the datas
    requirements : 
        - df must only contain normalized numerical datas useful for the prediction
            AND a column of '1' must be added at index 0 for the interception
        - weights must contain the weights calculated for each variable 
            + the interception at index 0 
    """

    if len(weights) != len(dfLine) :
        raise ValueError("The number of weights must be equal to the number")
    if dfLine[0] != 1:
        raise ValueError("The first column of the datas must be '1' for product with interception")
    thetaTx = np.dot(weights, dfLine)
    return sigmoid(thetaTx)


functionMap = {
    "sigmoid" : sigmoid,
    "relu" : relu,
    "softmax" : softmax
}

lossMap = {
    "binaryCrossEntropy" : binaryCrossEntropy
}


def forwardPropagation(newWeights, biases, batchData, activationsByLayer) :
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