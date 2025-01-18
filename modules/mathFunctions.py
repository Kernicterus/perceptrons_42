import numpy as np

def sigmoid(Z):
    """
    Function that calculates the sigmoid of a given value
    alias g(z) and returns it
    """    
    return 1 / (1 + np.exp(-Z))


def relu(Z) :
    """
    Function that calculates the RELU of a given vector z
    and returns it
    """   
    return np.maximum(0, Z)


def reluDerivative(Z) :
    return (Z > 0).astype(float)


def softmax(Z):
    """
    Computes softmax for each column in a batch matrix z.
    """
    zMax = np.max(Z, axis=0, keepdims=True)
    zStable = Z - zMax
    exponentiation = np.exp(zStable)
    sumExp = np.sum(exponentiation, axis=0, keepdims=True)
    return exponentiation / sumExp


def sigmoidDerivative(Z) :
    return sigmoid(Z) * (1 - sigmoid(Z))
    

def binaryCrossEntropy(yPredicted, yTrueResults) :
    # yPredicted = np.array([[0.9, 0.2], [0.8, 0.4]])
    # yTrueResults = np.array([[1, 0], [1, 0]])
    # result : 0.5312

    if yPredicted.shape != yTrueResults.shape :
        raise ValueError("yPredicted and yResults does not have the same size")
    if not np.all((yTrueResults == 0) | (yTrueResults == 1)):
        raise ValueError("yTrueResults should only contain 0 or 1")
    epsilon = 1e-15
    yPredicted = np.clip(yPredicted, epsilon, 1 - epsilon)
    m = len(yPredicted)

    array = []
    for entry in range(yPredicted.shape[1]) :
        errorEntry = 0
        for k in range(yPredicted.shape[0]) :
            errorEntry += (np.log(yPredicted[k, entry]) if yTrueResults[k, entry] == 1 else np.log(1 - yPredicted[k, entry]))
        array.append(errorEntry)


    # array = [np.log(yPredicted[x, i]) if yTrueResults[x, i] == 1 else np.log(1 - yPredicted[x, i])
    # for x in range(yPredicted.shape[0])
    # for i in range(yPredicted.shape[1])]
    # lossBatch = - np.mean(yTrueResults * np.log(yPredicted) + (1 - yTrueResults) * np.log(1 - yPredicted))
    lossBatch = - 1 / m * np.sum(array)
    return lossBatch
    

def bCrossEntrDerivBias() :
    pass


def bCrossEntrDerivative(yPredicted, yTrueResults):
    