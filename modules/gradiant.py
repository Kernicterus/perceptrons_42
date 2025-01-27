import numpy as np
import pandas as pd
import modules.mathFunctions as fct

BETA = 0.9

functionMap = {
    "sigmoid" : fct.sigmoid,
    "relu" : fct.relu,
    "softmax" : fct.softmax,
    "leakyRelu" : fct.leakyRelu,
    "tanh" : fct.tanh
}

lossMap = {
    "binaryCrossEntropy" : fct.binaryCrossEntropy
}

partialDerivativeMap = {
    "sigmoid" : fct.sigmoidDerivative,
    "relu"    : fct.reluDerivative,
    "leakyRelu" : fct.leakyReluDerivative,
    "tanh" : fct.tanhDerivative
}

def getDeltaOutputLayer(activation, loss, yRealResults, yPredicted) :
    """
    Function to get the delta of the output layer regarding the activation and loss functions for simplification matters
    Parameters :
    - activation : the activation function
    - loss : the loss function
    - yRealResults : the real results
    - yPredicted : the predicted results
    Return : np.ndarray containing the delta
    """
    if (activation == "sigmoid" or activation == "softmax") and loss == "binaryCrossEntropy" :
        return yPredicted - yRealResults
    # elif activation == "sigmoid" and loss == "mse" :
    #     pass
    # elif activation == "softmax" and loss == "mse" :
    #     pass
    # elif activation == "tanh" and loss == "binaryCrossEntropy" :
    #     pass
    # elif activation == "tanh" and loss == "mse" :
    #     pass
    # elif activation == "relu" and loss == "binaryCrossEntropy" :
    #     pass
    else :
        raise ValueError("no activation/loss combination found")


def forwardPropagation(weights : dict[np.ndarray], biases : dict, batchData, activationsByLayer : list) -> dict[np.ndarray]:
    """
    Function to do the forward propagation of the network
    Parameters :
    - weights : the weights of the network
    - biases : the biases of the network
    - batchData : a batch of data to process
    - activationsByLayer : the activations functions by layer
    Return : the caches containing the Zb (no activated) and A (activated) values
    """

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
        cacheZb[idZ] = np.dot(weights[f"l{i + 1}"], inputValues) + biases[f"l{i + 1}"]
        try :
            cacheA[idZ] = functionMap[activationsByLayer[idAct]](cacheZb[idZ])
        except KeyError :
            raise ValueError(f"activation function '{activationsByLayer[idAct]}' not found")

    caches = {
        "Zb" : cacheZb,
        "A" : cacheA
    }
    return caches


def getAllDeltas(yRealResults : np.ndarray, caches : dict[np.ndarray], weights : list[np.ndarray], activationByLayer : list, lossFct : str) -> dict[np.ndarray] :
    nLayer = len(activationByLayer)
    deltas = {}
    for layer in range(nLayer - 1, 0, -1) :
        if layer == nLayer - 1 :
            deltas[f"l{layer}"] = getDeltaOutputLayer(activationByLayer[layer], lossFct, yRealResults, caches["A"][f"l{layer}"])
        else : 
            deltas[f"l{layer}"] = (partialDerivativeMap[activationByLayer[layer]](caches["A"][f"l{layer}"])) * np.dot(np.transpose(weights[f"l{layer + 1}"]), deltas[f"l{layer + 1}"])
    return deltas


def backwardPropagation(yRealResults : np.ndarray, caches : dict[np.ndarray], weights : list[np.ndarray],
                        activationByLayer : list, lossFct : str, learningRate, biases : dict, vW, vB) -> list[np.ndarray]:
    """
    Function to do the backward propagation of the network. In the first part, it calculates the errors and 
    in the second part, updates the weights and biases of the network by taking the errors and the 
    learning rate into account.
    Returns the new weights and biases
    """
    # errors calculations
    deltas = getAllDeltas(yRealResults, caches, weights, activationByLayer, lossFct)

    # weights upd
    newWeights = {}
    batchSize = len(yRealResults[0])

    for id, key in enumerate(weights) :
        deltaW = np.dot(deltas[f"l{id + 1}"], np.transpose(caches["A"][f"l{id}"])) / batchSize
        deltaB = np.sum(deltas[f"l{id + 1}"], axis=1, keepdims=True) / batchSize
        
        # gradiant descent basic
        # biases[f"l{id + 1}"] = biases[f"l{id + 1}"] - learningRate * deltaB
        # array = weights[key] - learningRate * deltaW
        # newWeights[f"l{id + 1}"] = (array)

        # Nesterov Accelerated Gradient
        vW[key] = BETA * vW[key] + learningRate * deltaW
        vB[key] = BETA * vB[key] + learningRate * deltaB

        newWeights[f"l{id + 1}"] = weights[key] - vW[key]
        biases[f"l{id + 1}"] = biases[f"l{id + 1}"] - vB[key]
    return newWeights, biases, vW, vB
