import dstools as dst
import json
import numpy as np
import pandas as pd
import sys
import os


def checkArgs(argv) :
    return True


def main() :
    try :
        if checkArgs(sys.argv) == False:
            return 1
        
        # step 1 : load the dataset
        rawDatas = dst.loadCsvToDf(sys.argv[1])
        # step 2 : drop the rows with missing values in target column and delete index column
        datas = dst.prepareCsv(rawDatas)
        # step 3 : extraction, numerization, filling missing values (MEDIAN) and  standardization of numerical datas
        normalizedDatas, numDatasParams = dst.extractAndPrepareNumericalDatas(datas)
        print(normalizedDatas)

        # step 4 : extraction, numerization, filling missing values (MEAN) and standardization of discrete datas

        # step 5 : regroup the datas and add the intercept
        
        # step 6: rename the columns of the dataframe with numerical indexes

        # step 7 : prepare the results for each classifier (0 or 1) : one vs all technique

        # step 8 : calculate the weights for each classifier

        # step 9 : save the weights and the parameters
    except Exception as e :
        print(f"Error : {e}")


if __name__ == "__main__" :
    main()
