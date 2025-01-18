import modules.dstools as dst
import modules.weights as w
import modules.gradiant as grd
import numpy as np
import sys
import os

def fileCheck(file, extension) -> bool:
    if file.split('.')[-1] != extension:
        print(f"Error: wrong {file} : not {extension} extension")
        return False
    if not os.path.exists(file):
        print(f"Error: {file} not found")
        return False
    if not open(file, 'r').readable():
        print(f"Error: {file} not readable")
        return False
    if open(file, 'r').read() == "":
        print("Error: empty file")
        return False
    return True


def checkArgs(args) -> bool :
    if len(args) != 3:
        print("Error: wrong number of arguments")
        return False
    if fileCheck(args[1], "csv") == False :
        return False
    if fileCheck(args[2], "json") == False :
        return False
    return True


def launchTraining(weights : list[np.ndarray], model : dict, normalizedDatas : np.ndarray, yRealResults : np.ndarray, biases : list) -> list[np.ndarray]:
    learningRate = model["model_fit"]["learning_rate"] 
    batchSize = model["model_fit"]["batch_size"] 
    epochs = model["model_fit"]["epochs"] 
    lossFct = model["model_fit"]["loss"]
    activationByLayer = dst.getActivations(model)
    newWeights = weights
    for i in range(epochs):
        print(f"epoch {i}")
        for j in range(0, len(normalizedDatas), batchSize) :
            batchData = np.transpose(normalizedDatas[j:j + batchSize])
            caches = grd.forwardPropagation(newWeights, biases, batchData, activationByLayer)
            newWeights, biases = grd.backwardPropagation(yRealResults[:, j:j + batchSize], caches, newWeights, activationByLayer, lossFct, learningRate, biases)
    return newWeights, biases


def main() :
    try :
        if checkArgs(sys.argv) == False:
            print("Usage : train.py <dataset.csv> <model.json>")
            return 1
        
        # step 1 : load the dataset
        rawDatas = dst.loadCsvToDf(sys.argv[1])

        # step 2 : drop the rows with missing values in target column and delete index column
        datas = dst.prepareCsv(rawDatas)

        # step 3 : extraction, numerization, filling missing values (MEDIAN) and  standardization of numerical datas
        normalizedDatas, numDatasParams = dst.extractAndPrepareNumericalDatas(datas)

        # step 4 : extract and prepare the results : M=1, B=0
        binaryResultsByClasses = dst.targetBinarization(datas['f1'])

        # step 5 : load the json network architecture
        model = dst.loadJson(sys.argv[2])

        # step 6 : build the weight matrices + initialization
        weights = w.weightsInit(normalizedDatas, binaryResultsByClasses, model)

        # step 6b : prepare the bias
        biases = [np.full((weights[i].shape[0], 1), 0.001) for i in range(len(weights))]

        # step 7 : gradiant descent
        normalizedDatasNp = normalizedDatas.to_numpy()
        dst.saveCsv("dataNormalized.csv", normalizedDatasNp)
        weights = launchTraining(weights, model, normalizedDatasNp, binaryResultsByClasses, biases)

        # step 8 :

        # step 9 : save the weights and the parameters
    except Exception as e :
        print(f"Error : {e}")
        # raise Exception(f"Error : {e}")


if __name__ == "__main__" :
    main()
