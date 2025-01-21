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


def forwardPropagation(weights : list[np.ndarray], biases : list, batchData, activationsByLayer : list) -> dict[np.ndarray]:
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
        # print(i)
        # print(activationsByLayer[idAct])
        print(f"layer {i} :shape weigths {weights[i].shape} - shape input {inputValues.shape}")
        print(f"biases[i] shape = {biases[i].shape}")
        cacheZb[idZ] = np.dot(weights[i], inputValues) + biases[i]
        print(cacheZb[idZ].shape)
        # print (f"cache Z l{i} + bias : {cacheZb[idZ]}")
        cacheA[idZ] = functionMap[activationsByLayer[idAct]](cacheZb[idZ])
        # if activationsByLayer[idAct] == "softmax" :
        #     print(f"cache Zb l{i+1} : {cacheZb[idZ]}")
        #     print(f"cache A l{i+1} : {cacheA[f"l{i + 1}"]}")
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
        print(layer)
        if layer == nLayer - 1 :
            deltas[f"l{layer}"] = getDeltaOutputLayer(activationByLayer[layer], lossFct, yRealResults, caches["A"][f"l{layer}"])
            # print(f"delta layer l{layer} : {deltas[f"l{layer}"]}")
            print(f"deltas l{layer} shape : {deltas[f'l{layer}'].shape}")
        else : 
            print(f"weights id[{layer}] shape : {weights[layer].shape}") # attention les indexs des poids sont inférieurs de 1 aux indexs des layers
            print(f"deltas l{layer + 1} shape : {deltas[f'l{layer + 1}'].shape}")
            print(f"activation fct : {activationByLayer[layer]}")
            M = np.dot(np.transpose(weights[layer]), deltas[f"l{layer + 1}"])
            print(f"M:{M}")
            deltas[f"l{layer}"] = (partialDerivativeMap[activationByLayer[layer]](caches["A"][f"l{layer}"])) * np.dot(np.transpose(weights[layer]), deltas[f"l{layer + 1}"])
            # print(f"delta layer l{layer} : {deltas[f"l{layer}"]}")
            print(f"delta layer l{layer} shape : {deltas[f"l{layer}"].shape}")
            print("----")
            # lastErr = errorsByLayer[f"layer{layer + 1}"]
            # loss = lastErr 
        # errorsByLayer[f"layer{layer}"] = loss

    # weights upd
    newWeights = []
    newBiases = []
    # newWeights = weights - learningRate * ​∂L/​∂w SOIT = ​∂L/​∂ypred * ​∂pred/​∂z *​ ∂z/​∂w SOIT ​∂L/​∂ypred * ​∂pred/​∂z * 
    print(f"len w array :{len(weights)}")
    batchSize = len(yRealResults[0])
    print(f"batchsize = {batchSize}")

    for id, array in enumerate(weights) :
        print(f"shape array : {array.shape}")
        print(f"shape deltas[l{id + 1}] : {deltas[f'l{id + 1}'].shape}")
        print(f"shape caches[A][l{id}] : {caches["A"][f"l{id}"].shape}")
        deltaW = np.dot(deltas[f"l{id + 1}"], np.transpose(caches["A"][f"l{id}"])) /batchSize
        array = array - learningRate * deltaW
        biases[id] = biases[id] - learningRate * deltas[f"l{id + 1}"]
        newWeights.append(array)
        newBiases.append(biases[id])
        print("-----")
    # newBiases = biases - learningRate * biasesGrd
    return newWeights, newBiases
