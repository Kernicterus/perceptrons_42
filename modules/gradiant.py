import numpy as np
import pandas as pd
import modules.mathFunctions as fct


functionMap = {
    "sigmoid" : fct.sigmoid,
    "relu" : fct.relu,
    "softmax" : fct.softmax
}

lossMap = {
    "binaryCrossEntropy" : fct.binaryCrossEntropy
}

partialDerivativeMap = {
    "binaryCrossEntropy" : fct.bCrossEntrDerivative,
    "sigmoid" : fct.sigmoidDerivative,
    "relu"    : fct.reluDerivative
    # "softmax" : fct.softmaxDerivative,
}

def getDeltaOutputLayer(activation, loss, yRealResults, yPredicted) :
    if (activation == "sigmoid" or activation == "softmax") and loss == "binaryCrossEntropy" :
        return yPredicted - yRealResults
    elif activation == "sigmoid" and loss == "mse" :
        pass
    elif activation == "softmax" and loss == "mse" :
        pass
    elif activation == "tanh" and loss == "binaryCrossEntropy" :
        pass
    elif activation == "tanh" and loss == "mse" :
        pass
    elif activation == "relu" and loss == "binaryCrossEntropy" :
        pass
    else :
        raise ValueError("no activation/loss combination found")


def forwardPropagation(weights : dict[np.ndarray], biases : list, batchData, activationsByLayer : list) -> dict[np.ndarray]:
    cacheZb = {}
    cacheA = {}
    for i in range(len(weights)) :
        idZ = f"l{i + 1}"
        idAct = i + 1
        if i == 0 :
            inputValues = batchData
            cacheA[f"l{i}"] = inputValues
        else :
            inputValues = cacheA[f"l{i}"]
        cacheZb[idZ] = np.dot(weights[f"l{i + 1}"], inputValues) + biases[i]
        cacheA[idZ] = functionMap[activationsByLayer[idAct]](cacheZb[idZ])

    caches = {
        "Zb" : cacheZb,
        "A" : cacheA
    }
    return caches


def backwardPropagation(yRealResults : np.ndarray, caches : dict[np.ndarray], weights : list[np.ndarray],
                        activationByLayer : list, lossFct, learningRate, biases) -> list[np.ndarray]:
    
    # errors calculations
    batchSize = len(yRealResults)
    nLayer = len(activationByLayer)
    deltas = {}
    for layer in range(nLayer - 1, 0, -1) :
        if layer == nLayer - 1 :
            deltas[f"l{layer}"] = getDeltaOutputLayer(activationByLayer[layer], lossFct, yRealResults, caches["A"][f"l{layer}"])
        else : 
            M = np.dot(np.transpose(weights[f"l{layer + 1}"]), deltas[f"l{layer + 1}"])
            deltas[f"l{layer}"] = (partialDerivativeMap[activationByLayer[layer]](caches["A"][f"l{layer}"])) * np.dot(np.transpose(weights[f"{layer + 1}"]), deltas[f"l{layer + 1}"])

    # weights upd
    newWeights = {}
    newBiases = []
    batchSize = len(yRealResults[0])

    for id, array in enumerate(weights) :
        deltaW = np.dot(deltas[f"l{id + 1}"], np.transpose(caches["A"][f"l{id}"])) /batchSize
        array = array - learningRate * deltaW
        biases[id] = biases[id] - learningRate / batchSize * np.sum(deltas[f"l{id + 1}"], axis=1, keepdims=True)
        newWeights[f"l{id + 1}"] = (array)
        newBiases.append(biases[id])
    return newWeights, newBiases
