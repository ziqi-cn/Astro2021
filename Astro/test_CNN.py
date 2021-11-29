import numpy as np
import pandas as pd
from numpy import matrix
from src import *
from sklearn.metrics import confusion_matrix
load_config("config/config.json")
logger.info("Config loaded.")

testset, testlabel = load_feather(Config['feather']['test'])
logger.info("Data set loaded.")

example1 = CNN()
model = example1.res18lm()

#test model 1
model.load_weights('models/model1/model')
y_pred = model.predict(testset,verbose=1)
y_pred=np.argmax(y_pred,axis=1)
df = pd.DataFrame(data=y_pred,columns=["pred"])
df.to_csv("pred_model1.csv")
matrix = confusion_matrix(testlabel, y_pred)
logger.info("Confusion Matrix of Model 1:\n{}".format(matrix))
