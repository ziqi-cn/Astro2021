import numpy
import multiprocessing
from multiprocessing.dummy import Pool
from .func import *
from .log import logger
from .process_bar import ShowProcess

class kNN:
    __dataSet = numpy.array([])
    __labels = []
    #多线程加速，线程数默认为CPU个数
    multiprocess_enabled = True;
    threads = multiprocessing.cpu_count()
    
    show_processbar = True
    
    #kNN分类器
    def kNNClassify(self, testData, k, distFunc = Euclid):
        #testData-测试数据;dataSet-训练数据集;labels-标签集;k-k值;distFunc-距离函数
        numSample = self.__dataSet.shape[0]
        distance = distFunc(numpy.tile(testData, (numSample, 1)), self.__dataSet)
        sortedDistIndices = numpy.argsort(distance)
        classCount = {}
        for i in range(k):
            voteLabel = self.__labels[sortedDistIndices[i]]
            classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
        maxCount = 0
        for key, value in classCount.items():  
            if value > maxCount:
                maxCount = value  
                maxIndex = key 
        return maxIndex
    
    #封装kNNClassify，用于并行化
    def __runClassify(self, args):
        return self.kNNClassify(args[0], args[1], args[2])
    
    #kNN测试
    def kNN(self, testSet, labels, k, distFunc = Euclid):
        numSample = min(testSet.shape[0], len(labels))
        error = 0
        TP = [0, 0, 0, 0]
        P = [0, 0, 0, 0]
        R = [0, 0, 0, 0]
        process_bar = ShowProcess(numSample)
        process_bar.visible = self.show_processbar
        if self.multiprocess_enabled:
            pool = Pool(self.threads)
            tasks = [(testData, k, distFunc) for testData in testSet]
            i = 0
            for label in pool.imap(self.__runClassify, tasks):
                P[label] += 1
                R[labels[i]] += 1
                if (label != labels[i]):
                    error += 1
                    logger.debug("Wrong result at index {}, predict: {}, expect: {}".format(i, label, labels[i]))
                else:
                    TP[label] += 1
                process_bar.show_process()
                i += 1
                if (i >= numSample):
                    break
            process_bar.close()
            errorRate = error / numSample
        else:
            for i in range(numSample):
                label = self.kNNClassify(testSet[i], k, distFunc)
                P[label] += 1
                R[labels[i]] += 1
                if (label != labels[i]):
                    error += 1
                    logger.debug("Error index: {}".format(i))
                else:
                    TP[label] += 1
                process_bar.show_process()
            process_bar.close()
            errorRate = error / numSample
        marco_f1 = 0
        for i in range(3):
            precision = TP[i]/P[i]
            recall = TP[i]/R[i]
            f1 = 2*precision*recall/(precision+recall)
            marco_f1 += f1
            #logger.info("F1 score of class {}: {}".format(i, f1))
        return errorRate, marco_f1/3.0
    
    def __init__(self, dataSet, labels):
        self.__dataSet, self.__labels = dataSet, labels
