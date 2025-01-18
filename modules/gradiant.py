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
    "binaryCrossEntropy" : {
        "bias" : fct.bCrossEntrDerivBias, 
        "coeff" : fct.bCrossEntrDerivCoeff
    }
}


def forwardPropagation(newWeights : list[np.ndarray], biases : list, batchData, activationsByLayer : list) -> dict[np.ndarray]:
    cacheZb = {}
    cacheA = {}
    for i in range(len(newWeights)) :
        idZ = f"l{i + 1}"
        idAct = i + 1
        if i == 0 :
            inputValues = batchData
        else :
            inputValues = cacheA[f"l{i}"]
        # print(i)
        # print(activationsByLayer[idAct])
        # print(f"layer {i} :shape weigths {newWeights[i].shape} - shape input {inputValues.shape}")
        cacheZb[idZ] = np.dot(newWeights[i], inputValues) + biases[i]
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
    nLayer = len(activationByLayer)
    errorsByLayer = {}
    for layer in range(nLayer - 1, 0, -1) :
        # print(activationByLayer[nLayer - layer - 1])
        if layer == nLayer - 1 :
            loss = lossMap[lossFct](caches["A"][f"l{layer}"], yRealResults)
        else : 
            lastErr = errorsByLayer[f"layer{layer + 1}"]
            loss = lastErr 
        errorsByLayer[f"layer{layer}"] = loss
        print(errorsByLayer[f"layer{layer}"])
        print(layer)

    print(errorsByLayer)
    weightGrd = 0
    biasesGrd = 0
    # weights upd
    # newWeights = weights - learningRate * ​∂L/​∂w SOIT = ​∂L/​∂ypred * ​∂pred/​∂z *​ ∂z/​∂w
    # newBiases = biases - learningRate * biasesGrd
    return newWeights, newBiases
