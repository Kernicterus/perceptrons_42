import modules.dstools as dst
import modules.weights as w
import modules.gradiant as grd
import modules.mathFunctions as mf
import modules.tools as tls
import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt

BETA = 0.9
PATIENCE = 10

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
    return True


def graphManager(graphDatas : np.ndarray) :
    epochs = range(1, len(graphDatas) + 1)
    losses = graphDatas[:, 0]
    valLosses = graphDatas[:, 1]
    accuracies = graphDatas[:, 2]
    accuraciesVal = graphDatas[:, 3]

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].plot(epochs, losses, color='blue', linestyle='-', label='train')
    axes[0].plot(epochs, valLosses, color = 'orange', linestyle=':', label='validation')
    axes[0].set_title("Loss evolution (BCE function)")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, accuracies, color='blue', linestyle='-', label='accuracy')
    axes[1].plot(epochs, accuraciesVal, color = 'orange', linestyle=':', label='validation accuracy')
    axes[1].set_title("Accuracy evolution")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("loss.png")
    # plt.show()


def launchTraining(weights : dict, biases : dict, model : dict, datas : dict) :
    learningRate = model["model_fit"]["learning_rate"] 
    batchSize = model["model_fit"]["batch_size"] 
    epochs = model["model_fit"]["epochs"] 
    lossFct = model["model_fit"]["loss"]

    activationByLayer = dst.getActivations(model[model["model_fit"]["network"]])
    newWeights = weights.copy()
    yPredicted = np.zeros(datas["yRealResultsTrain"].shape)
    graphDatas = np.zeros((epochs, 4))
    recordedWeights = []
    recordedBiases = []

    if datas["yRealResultsTrain"].shape[0] != 2:
        raise AssertionError("Only binary classification is supported")

    recordedWeights.append(newWeights)
    recordedBiases.append(biases)
    # for Nesterov Accelerated Gradient
    vW = {key: np.zeros_like(weights[key]) for key in weights}
    vB = {key: np.zeros_like(biases[key]) for key in biases}
    # ----
    patience = PATIENCE
    iMax = 0  
    for i in range(1, epochs + 1):
        for j in range(0, len(datas["normalizedDatas"]), batchSize) :
            aWeights = {key: newWeights[key] - BETA * vW[key] for key in newWeights}
            aBiases = {key: biases[key] - BETA * vB[key] for key in biases}
            batchData = np.transpose(datas["normalizedDatas"][j:j + batchSize])
            caches = grd.forwardPropagation(aWeights, aBiases, batchData, activationByLayer)
            yPredicted[:, j:j + batchSize] = caches["A"][f"l{len(aWeights)}"]
            newWeights, biases, vW, vB = grd.backwardPropagation(datas["yRealResultsTrain"][:, j:j + batchSize], caches, aWeights, activationByLayer, lossFct, learningRate, aBiases, vW, vB)
        recordedWeights.append(newWeights)
        recordedBiases.append(biases)
        yPredictedVal = tls.getPredictedValues(newWeights, biases, datas["normalizedDatasVal"], activationByLayer)
        
        loss, accuracy = tls.getEvaluationMetrics(yPredicted, datas["yRealResultsTrain"])
        valLoss, accuracyVal = tls.getEvaluationMetrics(yPredictedVal, datas["yRealResultsVal"])

        if i % 5 == 0 :
            print(f"epoch {i}/{epochs} - loss : {loss:.4f} - val_loss : {valLoss:.4f}")
            print(f"    accuracy : {accuracy:.4f} - val_accuracy : {accuracyVal:.4f}")
        
        graphDatas[i - 1] = [loss, valLoss, accuracy, accuracyVal]

        # early stopping
        if i == 1 :
            valLossMin = valLoss
        if valLoss > valLossMin :
            patience -= 1
            if patience == 0 :
                graphDatas = graphDatas[:i]
                newWeights = recordedWeights[iMax]
                biases = recordedBiases[iMax]
                metr = graphDatas[iMax - 1]
                print(f"Early stopping at epoch {i} - returing to state at epoch {iMax}")
                print(f"    epoch {iMax}/{epochs} - loss : {metr[0]:.4f} - val_loss : {metr[1]:.4f}")
                print(f"        accuracy : {metr[2]:.4f} - val_accuracy : {metr[3]:.4f}")
                break
        else :
            valLossMin = valLoss
            iMax = i
            patience = PATIENCE

    return newWeights, biases, graphDatas


def main() :
    try :
        if checkArgs(sys.argv) == False:
            print("Usage : train.py <dataset.csv> <validation dataset.csv> <model.json> <json output name> <target column name> ")
            return 1
        
        # step 1 : load and process the datas
        print("Loading datas...")
        datasForTraining = {}
        normalizedDatas, numDatasParams, binaryResultsByClasses, targetClasses = tls.loadDatas(sys.argv[1], sys.argv[5])
        normalizedDatasVal, numDatasParamsVal, binaryResultsByClassesVal, tc = tls.loadDatas(sys.argv[2], sys.argv[5])
        datasForTraining["normalizedDatas"] = normalizedDatas.to_numpy()
        datasForTraining["normalizedDatasVal"] = normalizedDatasVal.to_numpy()

        # step 2 : load the json network architecture
        model = dst.loadParseJsonNetwork(sys.argv[3])

        # step 3 : build the weight matrices + initialization
        print("Building weights...")
        weights = w.weightsInit(datasForTraining["normalizedDatas"], binaryResultsByClasses, model[model["model_fit"]["network"]])

        # step 4 : prepare the bias
        biases = {f"l{i + 1}": np.full((weights[f"l{i + 1}"].shape[0], 1), 0.001) for i in range(len(weights))}

        # step 5 : gradiant descent
        print(f"Training of model '{model['model_fit']['network']}' started")
        datasForTraining["yRealResultsTrain"] = binaryResultsByClasses
        datasForTraining["yRealResultsVal"] = binaryResultsByClassesVal
        weights, biases, graphDatas = launchTraining(weights, biases, model, datasForTraining)

        # step 8 : save the weights and the parameters
        dst.saveTrainingParameters(sys.argv[4], model, weights, biases, numDatasParams, sys.argv[5], targetClasses)
        print(f"Training ended successfully - model saved as {sys.argv[4]}.json")

        # step 9 : graph
        graphManager(graphDatas)

    except Exception as e :
        print(f"Error : {e}")
        raise Exception(f"Error : {e}")


if __name__ == "__main__" :
    main()
