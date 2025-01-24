import modules.dstools as dst
import modules.weights as w
import modules.gradiant as grd
import numpy as np
import sys
import os    

def main() :
    try :
        pass
        # caches = grd.forwardPropagation(newWeights, biases, np.transpose(normalizedDatas), activationByLayer)
        # print(caches["A"]["l5"])
        
    except Exception as e :
        print(f"Error : {e}")


if __name__ == "__main__" :
    main()

