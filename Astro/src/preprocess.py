import numpy
from scipy import signal
import random
import scipy.stats as st
from collections import Counter

def moving_average(data, n):
    #new_data = numpy.arange((numpy.size(data, 1)-n)*numpy.size(data, 0)).astype(numpy.float)
    #new_data = new_data.reshape(numpy.size(data, 0), numpy.size(data, 1)-n)
    new_data = numpy.arange(data.size).astype(numpy.float)
    new_data = new_data.reshape(numpy.size(data, 0), numpy.size(data, 1))
    for i in range(numpy.size(data, 0)):
        new_data[i] = numpy.convolve(data[i], numpy.ones(n)/float(n), "same")
    return new_data

def normalize(data):
    new_data = numpy.arange(data.size).astype(numpy.float)
    new_data = new_data.reshape(numpy.size(data, 0), numpy.size(data, 1))
    for i in range(numpy.size(data, 0)):
        norm = numpy.linalg.norm(data[i])
        if norm != 0:
            new_data[i] = data[i] / norm
    return new_data

def unitization(data):
    new_data = numpy.arange(data.size).astype(numpy.float)
    new_data = new_data.reshape(numpy.size(data, 0), numpy.size(data, 1))
    for i in range(numpy.size(data, 0)):
        max = numpy.max(data[i])
        if max != 0:
            new_data[i] = data[i] / max
    return new_data

def maxmin(data):
    new_data = numpy.arange(data.size).astype(numpy.float)
    new_data = new_data.reshape(numpy.size(data, 0), numpy.size(data, 1))
    for i in range(numpy.size(data, 0)):
        max = numpy.max(data[i])
        min = numpy.min(data[i])
        if max != min:
            new_data[i] = (data[i]-min) / (max-min)
        else:
            new_data[i] = data[i] - max + 0.5
    return new_data

def stats_info(data):
    row = numpy.size(data, 0)
    a = numpy.zeros(shape=(row, 2), dtype='float32')
    a[:,0] = data.mean(axis=1)
    a[:,1] = numpy.linalg.norm(data, axis=1)
    return a

def black_info(data):
    a = 1.19e-16
    b = 1.44e-2
    le = [i for i in range(2600)]
    le = numpy.array(le)
    le = le * 0.2 + 380.0
    le = le * 1e-9
    y = - b / le
    row = numpy.size(data, 0)
    nd = numpy.zeros(shape=(row, 5), dtype='float32')
    for i in range(row):
        x = data[i,:2600]
        #f, e = signal.butter(8, 0.05, 'lowpass')
        #x =  signal.filtfilt(f, e, x)
        x[x <= 0] = 1e-3
        x = x * le**5 / a
        x = numpy.log(x)
        nd[i,0], nd[i,1], nd[i,2], nd[i,3], nd[i,4] = st.linregress(x, y)
    return nd

def add_noise(data, label):
    c = Counter(label)
    a = numpy.zeros(shape=(c[1]*4+c[2]*15, 2600), dtype='float32')
    l = []
    p = 0
    for i in numpy.c_[data, label]:
        if i[-1] == 1:
            for j in range(4):
                a[p] = i[0:2600] + numpy.random.normal(0, 2.1, 2600)
                p+=1
                l += [1]
        elif i[-1] == 2:
            for j in range(15):
                a[p] = i[0:2600] + numpy.random.normal(0, 2.1, 2600)
                p+=1
                l += [2]
    l = numpy.array(l)
    return a, l
            
        


