import numpy
import pandas
import feather
from scipy.stats import zscore
from src import *

load_config()
logger.info("Config loaded.")

trainset, trainlabel = load_feather("data/feather/train.feather")
testset, testlabel = load_feather("data/feather/test.feather")
logger.info("Data set loaded.")

d, l = add_noise(trainset, trainlabel)

trainset = numpy.append(trainset, d, axis=0)
trainlabel = numpy.append(trainlabel, l)
traindata = numpy.c_[trainset, trainlabel]
testdata = numpy.c_[testset, testlabel]

c = [str(i) for i in range(2601)]
f1 = pandas.DataFrame(data=traindata,columns=c)
f2 = pandas.DataFrame(data=testdata,columns=c)
logger.info("Data set added noise and expanded.")
f1.to_feather('data/feather/train_ex.feather')
f2.to_feather('data/feather/test_ex.feather')
logger.info("Data set stored to feather.")

d1 = stats_info(trainset)
b1 = black_info(trainset)
ntrain = normalize(trainset)
ntrain = numpy.c_[ntrain, d1]
ntrain = numpy.c_[ntrain, b1]

d2 = stats_info(testset)
b2 = black_info(testset)
ntest = normalize(testset)
ntest = numpy.c_[ntest, d2]
ntest = numpy.c_[ntest, b2]

traindata = numpy.c_[ntrain, trainlabel]
testdata = numpy.c_[ntest, testlabel]

c = [str(i) for i in range(2608)]
f1 = pandas.DataFrame(data=traindata,columns=c)
f2 = pandas.DataFrame(data=testdata,columns=c)
logger.info("Data set appended the mean and std.")
f1.to_feather('data/feather/train_norm.feather')
f2.to_feather('data/feather/test_norm.feather')
logger.info("Data set stored to feather.")
