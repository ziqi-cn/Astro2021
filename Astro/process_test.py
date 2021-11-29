import numpy
import pandas
import feather
from src import *
from collections import Counter
load_config("config/test.json")
logger.info("Config loaded.")

data, id = load_data()
logger.info("Data set loaded.")
logger.info("Data Size: {}".format(data.size))
logger.info("Data Rows: {}".format(numpy.size(data, 0)))
logger.info("Data Cols: {}".format(numpy.size(data, 1)))

testlabel = load_label(Config["testlabels"], id)
logger.info("Labels loaded.")
logger.info("Test Label Size: {}".format(len(testlabel)))
cnt = Counter(testlabel)
logger.debug(cnt)
logger.info("Include star: {}, galaxy: {}, qso: {}".format(cnt[0], cnt[1],  cnt[2]))

testlabel = numpy.array(testlabel)
testdata = numpy.c_[data, testlabel]

c = [str(i) for i in range(2601)]
df = pandas.DataFrame(data=testdata,columns=c)

df.to_feather('D:\\test\\test.feather')
logger.info("Data set stored to feather.")

d1 = stats_info(data)
b1 = black_info(data)
data = normalize(data)
data = numpy.c_[data, d1]
data = numpy.c_[data, b1]

testdata = numpy.c_[data, testlabel]

c = [str(i) for i in range(2608)]
df = pandas.DataFrame(data=testdata,columns=c)

logger.info("Data set appended additional infos.")
df.to_feather('D:\\test\\test_norm.feather')
logger.info("Data set stored to feather.")