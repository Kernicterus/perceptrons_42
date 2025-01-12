import dstools as dst
import json
import numpy as np
import pandas as pd
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


def nbNeuronsCalculation(stdDatas : pd.DataFrame, realResults : pd.Series, network : dict) -> np.array :
    nbNeurons = np.array([len(stdDatas.columns)])
    i = 0
    for item in network :
        if item.startswith("hidden_layer") :
            i += 1
            nbNeurons = np.append(nbNeurons, network[f"hidden_layer_{i}"]["neurons"])
    nbNeurons = np.append(nbNeurons, realResults.nunique())
    return nbNeurons


def weightsInit(stdDatas : pd.DataFrame, realResults : pd.Series, model : dict) :
    network = model[model["model_fit"]["network"]]
    neuronsByLayer = nbNeuronsCalculation(stdDatas, realResults, network)
    # weigths = np.array([x * y for x in neuronsByLayer for y in neuronsByLayer])

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

        # step 4 : extract and prepare the results
        binaryResults = dst.targetBinarization(datas['f1'])

        # step 5 : load the json network architecture
        model = dst.loadJson(sys.argv[2])

        # step 6 : build the weight matrices + initialization
        weights = weightsInit(normalizedDatas, binaryResults, model)
        # step 7 :

        # step 8 :

        # step 9 : save the weights and the parameters
    except Exception as e :
        print(f"Error : {e}")


if __name__ == "__main__" :
    main()
