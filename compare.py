import modules.dstools as dst
import modules.weights as w
import modules.gradiant as grd
import modules.tools as tls
import numpy as np
import sys
import matplotlib.pyplot as plt


def checkArgs(args) -> bool :
    if len(args) != 5:
        print("Error: wrong number of arguments")
        return False
    if dst.fileCheck(args[1], "csv") == False :
        return False
    if dst.fileCheck(args[2], "csv") == False :
        return False
    if dst.fileCheck(args[3], "json") == False :
        return False
    if not sys.argv[4].isalnum():
        print("Error: wrong output name, only alphanumeric characters allowed")
        return False
    return True


def graphManager(graphDatas : dict, epochs) :
    epochs = range(0, epochs + 1)
    for key in graphDatas:
        plt.plot(epochs, graphDatas[key], linestyle='-', label=f"model_{key}")

    plt.title("Accuracy evolution for different model")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig("modelComparaison.png")
    # plt.show()


def launchTraining(weights : dict[np.ndarray], biases : dict, model : dict, datas : dict, networkKey) :
    learningRate = model["model_fit"]["learning_rate"] 
    batchSize = model["model_fit"]["batch_size"] 
    epochs = model["model_fit"]["epochs"] 
    lossFct = model["model_fit"]["loss"]

    activationByLayer = dst.getActivations(model[networkKey])
    newWeights = weights.copy()
    yPredicted = np.zeros(datas["yRealResultsTrain"].shape)
    accuracyVal = []
    if datas["yRealResultsTrain"].shape[0] != 2:
        raise AssertionError("Only binary classification is supported")

    yPredictedVal = tls.getPredictedValues(newWeights, biases, datas["normalizedDatasVal"], activationByLayer)
    valLoss, accuracy = tls.getEvaluationMetrics(yPredictedVal, datas["yRealResultsVal"])
    accuracyVal.append(accuracy)
    print(f"*** Training model {networkKey} ***")
    for i in range(1, epochs + 1):
        for j in range(0, len(datas["normalizedDatas"]), batchSize) :
            batchData = np.transpose(datas["normalizedDatas"][j:j + batchSize])
            caches = grd.forwardPropagation(newWeights, biases, batchData, activationByLayer)
            yPredicted[:, j:j + batchSize] = caches["A"][f"l{len(newWeights)}"]
            newWeights, biases = grd.backwardPropagation(datas["yRealResultsTrain"][:, j:j + batchSize], caches, newWeights, activationByLayer, lossFct, learningRate, biases)
       
        yPredictedVal = tls.getPredictedValues(newWeights, biases, datas["normalizedDatasVal"], activationByLayer)
        
        valLoss, accuracy = tls.getEvaluationMetrics(yPredictedVal, datas["yRealResultsVal"])
        accuracyVal.append(accuracy)
    return accuracyVal


def main() :
    try :
        if checkArgs(sys.argv) == False:
            print("Usage : train.py <dataset.csv> <validation dataset.csv> <model.json> <target column name> ")
            return 1
        
        # step 1 : load and process the datas
        normalizedDatas, numDatasParams, binaryResultsByClasses, targetClasses = tls.loadDatas(sys.argv[1], sys.argv[4])
        normalizedDatasVal, numDatasParamsVal, binaryResultsByClassesVal, tc = tls.loadDatas(sys.argv[2], sys.argv[4])

        # step 2 : load the json network architecture
        model = dst.loadParseJsonNetwork(sys.argv[3])

        # step 3 : build the weight matrices + initialization for each model
        accuraciesDatas = {}
        datasForTraining = {}
        datasForTraining["normalizedDatas"] = normalizedDatas.to_numpy()
        datasForTraining["normalizedDatasVal"] = normalizedDatasVal.to_numpy()
        datasForTraining["yRealResultsTrain"] = binaryResultsByClasses
        datasForTraining["yRealResultsVal"] = binaryResultsByClassesVal
        for key in model :
            if key.startswith("network") :
                weights = w.weightsInit(normalizedDatas, binaryResultsByClasses, model[key])
                biases = {f"l{i + 1}": np.full((weights[f"l{i + 1}"].shape[0], 1), 0.001) for i in range(len(weights))}
                accuraciesDatas[key] = launchTraining(weights, biases, model, datasForTraining, key)

        # step 6 : graph
        graphManager(accuraciesDatas, model["model_fit"]["epochs"])

    except Exception as e :
        print(f"Error : {e}")
        raise Exception(f"Error : {e}")


if __name__ == "__main__" :
    main()
