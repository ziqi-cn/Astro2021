import numpy
import pandas
import feather
from scipy.stats import zscore
from src import *

load_config()
logger.info("Config loaded.")

data, id = load_data()
logger.info("Data set loaded.")
logger.debug("Data Size: {}".format(data.size))
logger.debug("Data Rows: {}".format(numpy.size(data, 0)))
logger.debug("Data Cols: {}".format(numpy.size(data, 1)))

trainlabel = load_label(Config["trainlabels"], id)
testlabel = load_label(Config["testlabels"], id[len(trainlabel):])
logger.info("Labels loaded.")
logger.debug("Train Label Size: {}".format(len(trainlabel)))
logger.debug("Test Label Size: {}".format(len(testlabel)))

trainset = data[:len(trainlabel)]
testset = data[len(trainlabel):]

trainlabel = numpy.array(trainlabel)
testlabel = numpy.array(testlabel)

traindata = numpy.c_[trainset, trainlabel]
testdata = numpy.c_[testset, testlabel]

c = [str(i) for i in range(2601)]
f1 = pandas.DataFrame(data=traindata,columns=c)
f2 = pandas.DataFrame(data=testdata,columns=c)

f1.to_feather('data/feather/train.feather')
f2.to_feather('data/feather/test.feather')
logger.info("Data set stored to feather.")