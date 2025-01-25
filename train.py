import modules.dstools as dst
import modules.weights as w
import modules.gradiant as grd
import modules.mathFunctions as mf
import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt


def checkArgs(args) -> bool :
    if len(args) != 6:
        print("Error: wrong number of arguments")
        return False
    if dst.fileCheck(args[1], "csv") == False :
        return False
    if dst.fileCheck(args[2], "json") == False :
        return False
    if dst.fileCheck(args[5], "csv") == False :
        return False
    if not sys.argv[3].isalnum():
        print("Error: wrong output name, only alphanumeric characters allowed")
        return False
    return True


def graphManager(graphDatas : np.ndarray) :
    epochs = range(1, len(graphDatas) + 1)
    losses = graphDatas[:, 0]
    valLosses = graphDatas[:, 1]
    plt.title("Loss evolution (BCE function)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(epochs, losses, color='red', linestyle='-', label='train')
    plt.plot(epochs, valLosses, color = 'blue', linestyle=':', label='validation')
    plt.legend()
    plt.savefig("loss.png")
    # plt.show()


def launchTraining(weights : dict[np.ndarray], model : dict, normalizedDatas : np.ndarray, normalizedDatasVal : np.ndarray, yRealResults : np.ndarray, yRealResultsVal, biases : dict) :
    learningRate = model["model_fit"]["learning_rate"] 
    batchSize = model["model_fit"]["batch_size"] 
    epochs = model["model_fit"]["epochs"] 
    lossFct = model["model_fit"]["loss"]
    activationByLayer = dst.getActivations(model[model["model_fit"]["network"]])
    newWeights = weights
    yPredicted = np.zeros(yRealResults.shape)
    graphDatas = np.zeros((epochs, 2))
    for i in range(1, epochs + 1):
        for j in range(0, len(normalizedDatas), batchSize) :
            batchData = np.transpose(normalizedDatas[j:j + batchSize])
            caches = grd.forwardPropagation(newWeights, biases, batchData, activationByLayer)
            yPredicted[:, j:j + batchSize] = caches["A"][f"l{len(newWeights)}"]
            newWeights, biases = grd.backwardPropagation(yRealResults[:, j:j + batchSize], caches, newWeights, activationByLayer, lossFct, learningRate, biases)
        if yRealResults.shape[0] == 2:
            loss = mf.binaryCrossEntropy(yPredicted[0], yRealResults[0])
            datasVal = np.transpose(normalizedDatasVal)
            cachesVal = grd.forwardPropagation(newWeights, biases, datasVal, activationByLayer)
            yPredictedVal = cachesVal["A"][f"l{len(newWeights)}"]
            valLoss = mf.binaryCrossEntropy(yPredictedVal[0], yRealResultsVal[0])
            print(f"epoch {i}/{epochs} - loss : {loss:.4f} - val_loss : {valLoss:.4f}")
            graphDatas[i - 1] = [loss, valLoss]
        else :
            raise AssertionError("Only binary classification is supported")
    return newWeights, biases, graphDatas


def main() :
    try :
        if checkArgs(sys.argv) == False:
            print("Usage : train.py <dataset.csv> <model.json> <json output name> <target column name> <validation dataset.csv>")
            return 1
        
        # step 1 : load the dataset
        rawDatas = dst.loadCsvToDf(sys.argv[1])
        rawDatasVal = dst.loadCsvToDf(sys.argv[5])
        if not sys.argv[4] in rawDatas.columns :
            raise AssertionError(f"target column '{sys.argv[4]}' not found in the dataset")
        if not sys.argv[4] in rawDatasVal.columns :
            raise AssertionError(f"target column '{sys.argv[4]}' not found in the validation dataset")
        targetColumnName = sys.argv[4]
        
        # step 2 : drop the rows with missing values in target column and delete index column
        datas = dst.prepareCsv(rawDatas, targetColumnName)
        datasVal = dst.prepareCsv(rawDatasVal, targetColumnName)

        # step 3 : extraction, numerization, filling missing values (MEDIAN) and  standardization of numerical datas
        normalizedDatas, numDatasParams = dst.extractAndPrepareNumericalDatas(datas)
        normalizedDatasVal, numDatasParamsVal = dst.extractAndPrepareNumericalDatas(datasVal)

        # step 4 : extract and prepare the results : M=1, B=0
        binaryResultsByClasses, targetClasses = dst.targetBinarization(datas[targetColumnName])
        binaryResultsByClassesVal, targetClassesVal = dst.targetBinarization(datasVal[targetColumnName])

        # step 5 : load the json network architecture
        model = dst.loadParseJsonNetwork(sys.argv[2])

        # step 6 : build the weight matrices + initialization
        weights = w.weightsInit(normalizedDatas, binaryResultsByClasses, model)

        # step 6b : prepare the bias
        biases = {f"l{i + 1}": np.full((weights[f"l{i + 1}"].shape[0], 1), 0.001) for i in range(len(weights))}

        # step 7 : gradiant descent
        normalizedDatasNp = normalizedDatas.to_numpy()
        normalizedDatasValNp = normalizedDatasVal.to_numpy()
        weights, biases, graphDatas = launchTraining(weights, model, normalizedDatasNp, normalizedDatasValNp,binaryResultsByClasses, binaryResultsByClassesVal, biases)

        # step 8 : save the weights and the parameters
        dst.saveTrainingParameters(sys.argv[3], model, weights, biases, numDatasParams, targetColumnName, targetClasses)
        print(f"Training ended successfully - model saved as {sys.argv[3]}.json")

        # step 9 : graph
        graphManager(graphDatas)

    except Exception as e :
        print(f"Error : {e}")
        # raise Exception(f"Error : {e}")


if __name__ == "__main__" :
    main()
