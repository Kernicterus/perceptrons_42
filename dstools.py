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


def targetBinarization(results: pd.Series) -> pd.Series :
    """
    Function that binarize the results between 1 (='M') and 0 (='B')
    Return the binarized results
    """  
    binaryResults = results.apply(lambda x: 1 if x == 'M' else 0)
    return binaryResults

def sigmoid(z):
    """
    Function that calculates the sigmoid of a given value
    alias g(z) and returns it
    """    
    return 1 / (1 + np.exp(-z))


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