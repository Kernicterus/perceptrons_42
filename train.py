import modules.dstools as dst
import modules.weights as w
import modules.gradiant as grd
import numpy as np
import sys


def checkArgs(args) -> bool :
    if len(args) != 5:
        print("Error: wrong number of arguments")
        return False
    if dst.fileCheck(args[1], "csv") == False :
        return False
    if dst.fileCheck(args[2], "json") == False :
        return False
    if not sys.argv[3].isalnum():
        print("Error: wrong output name, only alphanumeric characters allowed")
        return False
    return True


def launchTraining(weights : dict[np.ndarray], model : dict, normalizedDatas : np.ndarray, yRealResults : np.ndarray, biases : dict) -> dict[np.ndarray]:
    learningRate = model["model_fit"]["learning_rate"] 
    batchSize = model["model_fit"]["batch_size"] 
    epochs = model["model_fit"]["epochs"] 
    lossFct = model["model_fit"]["loss"]
    activationByLayer = dst.getActivations(model[model["model_fit"]["network"]])
    newWeights = weights
    for i in range(epochs):
        print(f"epoch {i}")
        for j in range(0, len(normalizedDatas), batchSize) :
            batchData = np.transpose(normalizedDatas[j:j + batchSize])
            caches = grd.forwardPropagation(newWeights, biases, batchData, activationByLayer)
            newWeights, biases = grd.backwardPropagation(yRealResults[:, j:j + batchSize], caches, newWeights, activationByLayer, lossFct, learningRate, biases)
    
    # caches = grd.forwardPropagation(newWeights, biases, np.transpose(normalizedDatas), activationByLayer)
    # print(caches["A"]["l5"])
    return newWeights, biases


def main() :
    try :
        if checkArgs(sys.argv) == False:
            print("Usage : train.py <dataset.csv> <model.json> <json output name> <target column name>")
            return 1
        
        # step 1 : load the dataset
        rawDatas = dst.loadCsvToDf(sys.argv[1])
        if not sys.argv[4] in rawDatas.columns :
            raise AssertionError(f"target column '{sys.argv[4]}' not found in the dataset")
        targetColumnName = sys.argv[4]
        
        # step 2 : drop the rows with missing values in target column and delete index column
        datas = dst.prepareCsv(rawDatas)

        # step 3 : extraction, numerization, filling missing values (MEDIAN) and  standardization of numerical datas
        normalizedDatas, numDatasParams = dst.extractAndPrepareNumericalDatas(datas)

        # step 4 : extract and prepare the results : M=1, B=0
        binaryResultsByClasses, targetClasses = dst.targetBinarization(datas[targetColumnName])

        # step 5 : load the json network architecture
        model = dst.loadParseJsonNetwork(sys.argv[2])

        # step 6 : build the weight matrices + initialization
        weights = w.weightsInit(normalizedDatas, binaryResultsByClasses, model)

        # step 6b : prepare the bias
        biases = {f"l{i + 1}": np.full((weights[f"l{i + 1}"].shape[0], 1), 0.001) for i in range(len(weights))}

        # step 7 : gradiant descent
        normalizedDatasNp = normalizedDatas.to_numpy()
        weights, biases = launchTraining(weights, model, normalizedDatasNp, binaryResultsByClasses, biases)

        # step 8 : save the weights and the parameters
        dst.saveTrainingParameters(sys.argv[3], model, weights, biases, numDatasParams, targetColumnName, targetClasses)

    except Exception as e :
        print(f"Error : {e}")
        # raise Exception(f"Error : {e}")


if __name__ == "__main__" :
    main()
