import numpy as np
import pandas as pd

# Initialization functions
def heUniform(weightsLayerShape : np.ndarray) :
    interval = np.sqrt(6 / weightsLayerShape[1])
    return np.random.uniform(-interval, interval, weightsLayerShape)


def heNormal(weightsLayerShape : np.ndarray) :
    std = np.sqrt(2 / weightsLayerShape[1])
    return np.random.normal(0, std, weightsLayerShape)


# Weights creation
def nbNeuronsCalculation(stdDatas: pd.DataFrame, realResults : pd.Series, network : dict) -> list :
    nbNeurons = []
    nbNeurons.append(len(stdDatas.columns))
    i = 1
    while f"hidden_layer_{i}" in network:
        if "neurons" in network[f"hidden_layer_{i}"] :
            nbNeurons.append(network[f"hidden_layer_{i}"]["neurons"])
        else :
            raise AssertionError(f"'neurons' missing in 'hidden_layer_{i}'")
        i += 1
    nbNeurons.append(realResults.nunique())
    return nbNeurons


def getInitializations(network : dict) -> list:
    initTypes = []
    i = 1
    while f"hidden_layer_{i}" in network:
        if "weights_init" in network[f"hidden_layer_{i}"] :
            initTypes.append(network[f"hidden_layer_{i}"]["weights_init"])
        else : 
            raise AssertionError(f"'weights_init' missing in 'hidden_layer_{i}'")
        i += 1
    if "weights_init" in network[f"output_layer"] :
        initTypes.append(network[f"output_layer"]["weights_init"])
    else : 
        initTypes.append(network["output_layer"]["weights_init"])
    return initTypes
    

def getInitFunc(funcTitle : str) :
    if funcTitle == "heUniform" :
        return heUniform
    else :
        raise ValueError(f"no initialization called '{funcTitle}' found")
    

def weightsInit(stdDatas : pd.DataFrame, realResults : pd.Series, model : dict) -> list[np.ndarray]:
    if not model["model_fit"]["network"] in model:
        raise AssertionError(f"No '{model["model_fit"]["network"]}' in model")
    network = model[model["model_fit"]["network"]]
    neuronsByLayer = nbNeuronsCalculation(stdDatas, realResults, network)
    print(f"neurons by layer : {neuronsByLayer}")
    initTypeByLayer = getInitializations(network)
    weights = [np.zeros((neuronsByLayer[i], neuronsByLayer[i - 1])) for i in range(1, len(neuronsByLayer))]
    for id, array in enumerate(weights) :
        initFunc = getInitFunc(initTypeByLayer[id])
        array = initFunc(array.shape)
        weights[id] = array
    return weights