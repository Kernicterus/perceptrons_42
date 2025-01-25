import modules.dstools as dst
import modules.gradiant as grd
import modules.mathFunctions as mf
import numpy as np

def loadDatas(path : str, targetColumnName : str) -> dict :
        # step 1 : load the dataset
        rawDatas = dst.loadCsvToDf(path)

        if not targetColumnName in rawDatas.columns :
            raise AssertionError(f"target column '{targetColumnName}' not found in the dataset")
        
        # step 2 : drop the rows with missing values in target column and delete index column
        datas = dst.prepareCsv(rawDatas, targetColumnName)

        # step 3 : extraction, numerization, filling missing values (MEDIAN) and  standardization of numerical datas
        normalizedDatas, numDatasParams = dst.extractAndPrepareNumericalDatas(datas)

        # step 4 : extract and prepare the results : M=1, B=0
        binaryResultsByClasses, targetClasses = dst.targetBinarization(datas[targetColumnName])

        return normalizedDatas, numDatasParams, binaryResultsByClasses, targetClasses


def getEvaluationMetrics(yPredicted : np.ndarray, yRealResults : np.ndarray) -> tuple:
    """
    Function to get the evaluation metrics
    Parameters :
    - yPredicted : the predicted values
    - yRealResults : the real results
    Return : tuple containing the loss, the accuracy
    """
    loss = mf.binaryCrossEntropy(yPredicted[0], yRealResults[0])
    accuracy = np.mean((yPredicted[0] > 0.5) == yRealResults[0])
    return loss, accuracy


def getPredictedValues(weights : dict[np.ndarray], biases : dict, datas, activationByLayer : list) -> np.ndarray:
    """

    """
    datasT = np.transpose(datas)
    caches = grd.forwardPropagation(weights, biases, datasT, activationByLayer)
    yPredicted = caches["A"][f"l{len(weights)}"]
    return yPredicted
