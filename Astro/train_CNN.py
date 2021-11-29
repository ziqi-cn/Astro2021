import numpy as np
from numpy import matrix
from src import *
from sklearn.metrics import confusion_matrix
load_config("config/config.json")
logger.info("Config loaded.")

trainset, trainlabel = load_feather(Config['feather']['train'])
testset, testlabel = load_feather(Config['feather']['test'])
logger.info("Data set loaded.")

exam = CNN(128)

model = exam.res18lm("1")
model, history = exam.run(trainset, trainlabel, testset, testlabel, model, 
                        'categorical_crossentropy', 2e-4, 5)

model = exam.res18lm("2")
model.load_weights('models/1/model')
model, history = exam.run(trainset, trainlabel, testset, testlabel, model, 
                        'categorical_crossentropy', 2e-5, 5)
#save_(model, history, 2)