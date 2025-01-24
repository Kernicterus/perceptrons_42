import numpy as np
import pandas as pd

# Initialization functions
def heUniform(weightsLayerShape : np.ndarray) :
    """
    Function to initialize the weights of a layer with the He Unifrom initialization
    Parameters :
    - weightsLayerShape : shape of the weights
    Return : np.ndarray containing the initialized weights
    """
    interval = np.sqrt(6 / weightsLayerShape[1])
    return np.random.uniform(-interval, interval, weightsLayerShape)


def heNormal(weightsLayerShape : np.ndarray) :
    """
    Function to initialize the weights of a layer with the He Normal initialization
    Parameters :
    - weightsLayerShape : shape of the weights
    Return : np.ndarray containing the initialized weights
    """
    std = np.sqrt(2 / weightsLayerShape[1])
    return np.random.normal(0, std, weightsLayerShape)


# Weights creation
def nbNeuronsCalculation(stdDatas: pd.DataFrame, outputLayerLen : int, network : dict) -> dict :
    """
    Function to calculate the number of neurons in each layer
    Parameters :
    - stdDatas : the standardized datas
    - outputLayerLen : the number of neurons in the output layer
    - network : the network architecture
    """
    nbNeurons = {}
    nbNeurons["l0"] = (len(stdDatas.columns))
    i = 1
    while f"hidden_layer_{i}" in network:
        if "neurons" in network[f"hidden_layer_{i}"] :
            nbNeurons[f"l{i}"] = (network[f"hidden_layer_{i}"]["neurons"])
        else :
            raise AssertionError(f"'neurons' missing in 'hidden_layer_{i}'")
        i += 1
    nbNeurons[f"l{i}"] = outputLayerLen
    return nbNeurons


def getInitializations(network : dict) -> list:
    """
    Function to get the initialization functions for each layer
    Parameters :
    - network : the network architecture
    Return : list of initialization functions
    """
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
        initTypes.append("default")
    return initTypes
    

def getInitFunc(funcTitle : str) :
    """
    Function to get the initialization function of the weights of a layer
    Parameters :
    - funcTitle : the title of the function
    Return : the function
    """
    if funcTitle == "heUniform" or funcTitle == "default" :
        return heUniform
    elif funcTitle == "heNormal" :
        return heNormal
    else :
        raise ValueError(f"no initialization called '{funcTitle}' found")
    

def weightsInit(stdDatas : pd.DataFrame, realResults : pd.Series, model : dict) -> dict[np.ndarray]:
    """
    Function to initialize the weights of the network by using the proper initialization function of each layer
    Parameters :
    - stdDatas : the standardized datas
    - realResults : the real results
    - model : the model architecture
    Return : the weights initialized
    """
    network = model[model["model_fit"]["network"]]
    neuronsByLayer = nbNeuronsCalculation(stdDatas, len(realResults), network)
    initTypeByLayer = getInitializations(network)
    weights = {f"l{i}":np.zeros((neuronsByLayer[f"l{i}"], neuronsByLayer[f"l{i - 1}"])) for i in range(1, len(neuronsByLayer))}
    for id, key in enumerate(weights) :
        initFunc = getInitFunc(initTypeByLayer[id])
        array = initFunc(weights[key].shape)
        weights[f"l{id + 1}"] = array
    return weights