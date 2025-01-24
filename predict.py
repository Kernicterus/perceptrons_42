import modules.dstools as dst
import modules.weights as w
import modules.gradiant as grd
import numpy as np
import pandas as pd
import sys
import json

def checkArgs(args) -> bool :
    """
    Function to check the arguments
    Parameters :
    - args : the arguments
    Return : True if the arguments are correct, False otherwise
    """
    if len(args) != 4:
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

def loadNetworkArchitecture(path : str) -> dict:
    """
    Function to load the json file containing the network architecture and all parameters needed
    """
    try : 
        with open(path, "r") as file:
            networkArchitecture = json.load(file)
        requiredKeys = {"network", "weights", "biases", "dataParameters"}
        if not requiredKeys.issubset(networkArchitecture.keys()):
            raise AssertionError("one of the following field is missing : network, weights, biases, dataParameters")
    except Exception as e :
        raise AssertionError(f"loading json : {e}")
    return networkArchitecture


def checkNetworkArchitecture(networkArchitecture : dict, dfValidation : pd.DataFrame, targetClassesNb) :
    network = networkArchitecture["network"]
    for key in networkArchitecture["network"] :
        if "neurons" not in networkArchitecture["network"][key] :
            raise AssertionError(f"'neurons' missing in {key}")
        if "activation" not in networkArchitecture["network"][key] :
            raise AssertionError(f"'activation' missing in {key}")
    if network["input_layer"]["neurons"] != len(dfValidation.columns) :
        raise AssertionError(f"number of neurons in the input layer is not valid (should be '{len(dfValidation.columns)}')")
    if network["output_layer"]["neurons"] != targetClassesNb :
        raise AssertionError(f"number of neurons in the output layer is not valid (should be '{targetClassesNb}')")
    for id in range(1, len(network) - 1) :
        if f"hidden_layer_{id}" not in network :
            raise AssertionError(f"'hidden_layer_{id}' not found in the network")
        if network[f"hidden_layer_{id}"]["neurons"] != len(networkArchitecture["weights"][f"l{id}"]) :
            raise AssertionError(f"number of neurons in the hidden layer {id} is not valid regarding the weights (should be '{len(networkArchitecture["weights"][f"l{id}"])}')")
        
    for key in networkArchitecture["dataParameters"] :
        if key not in dfValidation.columns :
            raise AssertionError(f"'{key}' column not found in the validation dataset")

    for id, key in enumerate(network) :
        if key == "input_layer" :
            continue
        elif key == "output_layer" :
            if network[key]["neurons"] != len(networkArchitecture["biases"][f"l{id}"]) :
                raise AssertionError(f"number of biases in the output layer is not valid (should be '{len(networkArchitecture['biases'][f'l{id}'])}')")
        else :
            if network[key]["neurons"] != len(networkArchitecture["biases"][f"l{id}"]) :
                raise AssertionError(f"number of biases in the {key} {id} is not valid (should be '{len(networkArchitecture['biases'][f'l{id}'])}')")
       

def getPredictedValues(dfValidation : pd.DataFrame, networkArchitecture : dict) :
        weights = {}
        biases = {}
        for key in networkArchitecture["weights"] :
            weights[key] = np.array(networkArchitecture["weights"][key])
            biases[key] = np.array(networkArchitecture["biases"][key])
        normalizedDatasNp = dfValidation.to_numpy()
        activationByLayer = dst.getActivations(networkArchitecture["network"])
        caches = grd.forwardPropagation(weights, biases, np.transpose(normalizedDatasNp), activationByLayer)
        return caches["A"][f"l{len(weights)}"]    

def formattingResults(predictedResults : np.ndarray, targetClasses) -> pd.Series :
    """
    Function to format the predicted results
    Parameters :
    - predictedResults : the predicted results
    - targetClasses : the target classes
    Return : the formatted results
    """

    targetClasses.sort()
    formattedResults = []
    classResults = np.argmax(predictedResults, axis=0)
    for result in classResults :
        formattedResults.append(targetClasses[result])
    return pd.Series(formattedResults)

def main() :
    try :
        # Step 0 : check the arguments
        if checkArgs(sys.argv) == False:
            raise AssertionError("Usage : predict.py <validation_dataset.csv> <modelParameters.json> <csv output name>")
        
        # Step 1 : load  the validation csv file
        dfValidation = dst.loadCsvToDf(sys.argv[1])

        # Step 2 : load  the json architecture file
        networkArchitecture = loadNetworkArchitecture(sys.argv[2])

        # Step 3 : create the results predicted dataframe and empty the second column
        targetColumnName = networkArchitecture["targetColumnName"]
        results = dfValidation.iloc[:, :2]
        targetClasses = networkArchitecture["targetClasses"]
        targetClassesNb = len(targetClasses)
        results.iloc[:, 1] = np.nan

        # Step 4 : check the datas of the network architecture
        dfValidation = dfValidation.iloc[:, 2:]
        checkNetworkArchitecture(networkArchitecture, dfValidation, targetClassesNb)

        # Step 5 : extract and standardization of the datas
        params = networkArchitecture["dataParameters"]
        for column in dfValidation.columns:
            dfValidation[column] = dst.normalizePdSeries(dfValidation[column], params[column])

        # Step 5 : compute de predicted values
        predictedResults = getPredictedValues(dfValidation, networkArchitecture)

        # Step 6 : format and save the predicted results into csv
        results[targetColumnName] = formattingResults(predictedResults, targetClasses)

        # Step 7 : save the results
        dst.saveCsv(f"{sys.argv[3]}.csv", results)

    except Exception as e :
        print(f"Error : {e}")
        raise Exception(f"Error : {e}")

if __name__ == "__main__" :
    main()

