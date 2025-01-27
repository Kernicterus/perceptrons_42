import modules.dstools as dst
import modules.weights as w
import modules.gradiant as grd
import modules.tools as tls
import numpy as np
import sys
import matplotlib.pyplot as plt

BETA = 0.9

def checkArgs(args) -> bool :
    if len(args) != 6:
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
    if sys.argv[5] not in ["network", "learning_rate"] :
        print("Error: wrong mode, only 'network' or 'learning_rate' allowed")
        return False
    return True


def graphManager(graphDatas : dict, epochs, outputName : str) :
    epochs = range(0, epochs + 1)
    for key in graphDatas:
        plt.plot(epochs, graphDatas[key], linestyle='-', label=f"model_{key}")

    plt.title("Accuracy evolution for different model")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig(outputName)
    # plt.show()


def launchTraining(weights : dict[np.ndarray], biases : dict, model : dict, datas : dict, networkKey, learningRate) -> list:
    batchSize = model["model_fit"]["batch_size"] 
    epochs = model["model_fit"]["epochs"] 
    lossFct = model["model_fit"]["loss"]

    activationByLayer = dst.getActivations(model[networkKey])
    newWeights = weights.copy()
    yPredicted = np.zeros(datas["yRealResultsTrain"].shape)
    accuracyVal = []

    if datas["yRealResultsTrain"].shape[0] != 2:
        raise AssertionError("Only binary classification is supported")

    vW = {key: np.zeros_like(weights[key]) for key in weights}
    vB = {key: np.zeros_like(biases[key]) for key in biases}

    yPredictedVal = tls.getPredictedValues(newWeights, biases, datas["normalizedDatasVal"], activationByLayer)
    valLoss, accuracy = tls.getEvaluationMetrics(yPredictedVal, datas["yRealResultsVal"])
    accuracyVal.append(accuracy)
    print(f"*** Training model {networkKey} ***")
    for i in range(1, epochs + 1):
        for j in range(0, len(datas["normalizedDatas"]), batchSize) :
            aWeights = {key: newWeights[key] - BETA * vW[key] for key in newWeights}
            aBiases = {key: biases[key] - BETA * vB[key] for key in biases}
            batchData = np.transpose(datas["normalizedDatas"][j:j + batchSize])
            caches = grd.forwardPropagation(aWeights, aBiases, batchData, activationByLayer)
            yPredicted[:, j:j + batchSize] = caches["A"][f"l{len(aWeights)}"]
            newWeights, biases, vW, vB = grd.backwardPropagation(datas["yRealResultsTrain"][:, j:j + batchSize], caches, aWeights, activationByLayer, lossFct, learningRate, aBiases, vW, vB)

        yPredictedVal = tls.getPredictedValues(newWeights, biases, datas["normalizedDatasVal"], activationByLayer)
        
        valLoss, accuracy = tls.getEvaluationMetrics(yPredictedVal, datas["yRealResultsVal"])
        accuracyVal.append(accuracy)
    return accuracyVal


def launchTrainingLearningRate(model : dict, datas : dict) :
    learningRates = [1, 0.1, 0.01, 0.001, 0.0001]
    accuraciesDatas = {}
    network = model[model["model_fit"]["network"]]
    for item in learningRates :
        weights = w.weightsInit(datas["normalizedDatas"], datas["yRealResultsTrain"], network)
        biases = {f"l{i + 1}": np.full((weights[f"l{i + 1}"].shape[0], 1), 0.001) for i in range(len(weights))}
        accuraciesDatas[f"{item}_LR"] = launchTraining(weights, biases, model, datas, model["model_fit"]["network"], item)

    # step 5 : graph
    graphManager(accuraciesDatas, model["model_fit"]["epochs"], "ModelComparisonLR.png")


def launchTrainingNetwork(model : dict, datas : dict) :
    accuraciesDatas = {}
    for key in model :
        if key.startswith("network") :
            weights = w.weightsInit(datas["normalizedDatas"], datas["yRealResultsTrain"], model[key])
            biases = {f"l{i + 1}": np.full((weights[f"l{i + 1}"].shape[0], 1), 0.001) for i in range(len(weights))}
            accuraciesDatas[key] = launchTraining(weights, biases, model, datas, key, model["model_fit"]["learning_rate"])

    # step 5 : graph
    graphManager(accuraciesDatas, model["model_fit"]["epochs"], "ModelComparisonNtw.png")


def main() :
    try :
        if checkArgs(sys.argv) == False:
            print("Usage : train.py <dataset.csv> <validation dataset.csv> <model.json> <target column name> <mode>")
            return 1
        
        # step 1 : load and process the datas
        normalizedDatas, numDatasParams, binaryResultsByClasses, targetClasses = tls.loadDatas(sys.argv[1], sys.argv[4])
        normalizedDatasVal, numDatasParamsVal, binaryResultsByClassesVal, tc = tls.loadDatas(sys.argv[2], sys.argv[4])
        datasForTraining = {}
        datasForTraining["normalizedDatas"] = normalizedDatas.to_numpy()
        datasForTraining["normalizedDatasVal"] = normalizedDatasVal.to_numpy()

        # step 2 : load the json network architecture
        model = dst.loadParseJsonNetwork(sys.argv[3])

        # step 3 : build the weight matrices + initialization for each model
        datasForTraining["yRealResultsTrain"] = binaryResultsByClasses
        datasForTraining["yRealResultsVal"] = binaryResultsByClassesVal

        # step 4 : training for each model
        if sys.argv[5] == "network" :
            launchTrainingNetwork(model, datasForTraining)
        else :
            launchTrainingLearningRate(model, datasForTraining)

    except Exception as e :
        print(f"Error : {e}")
        # raise Exception(f"Error : {e}")


if __name__ == "__main__" :
    main()
