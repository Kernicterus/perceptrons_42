import numpy as np
import pandas as pd
import json


def loadCsvToNp(path : str) -> np.ndarray :
    """
    Function to load a csv
    Parameters : path of the csv
    Return : np.ndarray containing the csv datass
    """
    try :
        data = np.genfromtxt(path, delimiter=",", dtype=None, encoding="utf-8")
        return data
    except Exception as e :
        raise Exception(e)


def loadCsvToDf(path: str) -> pd.DataFrame:
    """
    Function to load a csv
    Parameters : path of the csv
    Return : pd.DataFrame containing the csv datass
    """
    try:
        csv = pd.read_csv(path)
    except Exception as e:
        print(f"loading csv error : {e}")
        return None
    return csv


def loadJson(path : str) -> dict:
    """
    Function to load a json and returns it
    """
    try : 
        with open(path, "r") as file:
            data = json.load(file)
        required_keys = {"network", "loss", "learning_rate", "batch_size", "epochs"}
        if "model_fit" not in data :
            raise ValueError("model_fit field not found")
        else : 
            model = data["model_fit"]
            if not required_keys.issubset(model.keys()):
                raise AssertionError("one of the following field of 'model_fit' is missing : \
network, loss, learning_rate, batch_size, epochs")
    except Exception as e :
        raise AssertionError(f"loading json : {e}")
    return data


def saveCsv(outputName : str, datas : np.ndarray) :
    """
    Function to save a csv
    Parameters : 
    - path of the new file
    - the dataset
    """
    try : 
        df = pd.DataFrame(datas)
        df.to_csv(outputName, index=False)
    except Exception as e :
        raise Exception(e)


def randomlySplitCsv(datas : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to separate the dataset into two parts randomly chosen
    Parameters : a np.ndarray object containing the csv
    Return : 
    """ 
    numRows = datas.shape[0]
    shuffledIndexes = np.random.permutation(numRows)

    midpoint = numRows // 2
    part1Indexes = shuffledIndexes[:midpoint]
    part2Indexes = shuffledIndexes[midpoint:]

    part1 = datas[part1Indexes]
    part2 = datas[part2Indexes]
    return part1, part2


def prepareCsv(rawDatas : pd.DataFrame) -> pd.DataFrame :
    """
    Function to clean the dataset by deleting the index column and deleting rows with missing target
    Parameters : a pd.DataFrame object
    Return : a new pd.DataFrame containing the cleaned datas
    """ 
    datas = rawDatas.drop(columns="f0")
    datas["f1"] = datas["f1"].replace('', pd.NA)
    datasWithoutEmpty = datas.dropna(subset=['f1'])
    return datasWithoutEmpty


def normalizePdSeries(variable : pd.Series, parameters : pd.Series) -> pd.Series :
    """
    Function to standardize a given variable from its different values
    Parameters : a pd.Series object containing the mean and std of the variable
    Return : a new pd.Series containing the normalized values of the variable
    """ 
    variableNormalized = (variable - parameters['mean']) / parameters['std']
    return variableNormalized


def extractAndPrepareNumericalDatas(df : pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] :
    """
    Function that extract numerical datas, filled missing values with median and normalize datas
    Parameters : a pd.DataFrame object
    Return : a new dataFrame containing only numerical datas and a dataFrame containing 
    mean and std parameters for each variable
    """
    numericalDf = df.select_dtypes(include=['int64', 'float64'])
    parameters = pd.DataFrame(columns=numericalDf.columns, index=['mean', 'std', 'median'])
    for column in numericalDf.columns:
        median = numericalDf[column].median()
        mean = numericalDf[column].mean()
        std = numericalDf[column].std()
        numericalDf[column] = numericalDf[column].fillna(median)
        parameters[column] = [mean, std, median]
    for column in numericalDf.columns:
        numericalDf[column] = normalizePdSeries(numericalDf[column], parameters[column])
    return numericalDf, parameters


def targetBinarization(results: pd.Series) -> list[np.ndarray] :
    """
    Function that binarize the results between 1 and 0  for each class (one vs all)
    Return the binarized results for each class in a list of np array
    The classes are sorted by alphanumerical order
    """
    arrayList = []
    categories = pd.Categorical(results)
    categories.sort_values()
    for item in categories.categories :
        arrayList.append(results.apply(lambda x : 1 if x == item else 0 ).to_numpy())
    binaryResultsByClasses = np.vstack(arrayList)
    return binaryResultsByClasses


def getActivations(model : dict) :
    layers = model[model["model_fit"]["network"]]
    activations = []
    for layer, layerConfig in layers.items():
                if "activation" in layerConfig:
                    activations.append(layerConfig["activation"])
                else:
                    raise ValueError(f"The layer '{layer}' is missing the 'activation' key.")
    return activations