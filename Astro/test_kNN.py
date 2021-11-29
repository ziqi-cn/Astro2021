import numpy
from src import *

load_config()
logger.info("Config loaded.")

trainset, trainlabel = load_feather("data/feather/all/train.feather")
testset, testlabel = load_feather("data/feather/all/train.feather")
logger.info("Data set loaded.")
trainlabel = trainlabel.tolist()
testlabel = testlabel.tolist()

trainset = trainset[0:2600]
testset = testset[0:2600]

trainset = moving_average(trainset, 13)
testset = moving_average(trainset, 13)
l = [13*i+6 for i in range(200)]
trainset = trainset[:, l]
testset = testset[:, l]
trainset = maxmin(trainset)
testset = maxmin(testset)

example_1 = kNN(trainset, trainlabel)
logger.info("Classifier inited.")

rate, score = example_1.kNN(testset, testlabel, 5, Euclid)
logger.info("Done.")
logger.info("n = 13, Euclid, k = 5, Error Rate is {}, Marco F1 Score is {}".format(rate, score))