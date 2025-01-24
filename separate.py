import modules.dstools as dst
import sys

CSV_NAME_TRAIN = "data_train.csv"
CSV_NAME_TEST = "data_test.csv"


def main() :
    try : 
        if len(sys.argv) != 2 :
            raise Exception("Usage : separate.py <dataset>")
        rawDatas = dst.loadCsvToNp(sys.argv[1])
        datasTest, datasTrain = dst.randomlySplitCsv(rawDatas)
        print(f"CSV {sys.argv[1]} successfully splitted in two parts")
        dst.saveCsv(CSV_NAME_TEST, datasTest)
        print(f"CSV {CSV_NAME_TEST} successfully saved")
        dst.saveCsv(CSV_NAME_TRAIN, datasTrain)
        print(f"CSV {CSV_NAME_TRAIN} successfully saved")
    except Exception as e:
        print(f"Error : {e}")

if __name__ == "__main__" :
    main()
