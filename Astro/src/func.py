import numpy

#欧氏距离
def Euclid(x, y):
    squaredDiff = (x - y) ** 2
    distance = numpy.sum(squaredDiff, axis = 1).reshape(squaredDiff.shape[0]) ** 0.5
    return distance

#曼哈顿距离
def Manhattan(x, y):
    diffAbs = numpy.abs(x - y)
    distance = numpy.sum(diffAbs, axis = 1)
    return distance

#闵可夫斯基距离，p = 3
def LP3(x, y):
    squaredDiff = (x - y) ** 3
    squaredDiff = numpy.abs(squaredDiff)
    distance = numpy.sum(squaredDiff, axis = 1).reshape(squaredDiff.shape[0]) ** (1./3.)
    return distance

#闵可夫斯基距离，p = 4
def LP4(x, y):
    squaredDiff = (x - y) ** 4
    distance = numpy.sum(squaredDiff, axis = 1).reshape(squaredDiff.shape[0]) ** 0.25
    return distance

#普朗克函数
def Planck(x, T):
    h = 6.62607004e-34
    c = 299792458
    k = 1.380649e-23
    l = x * 1e-9
    I = 2*h*(c**2)/(l**5)*1/(numpy.exp(h*c/l/k/T)-1)
    return I

#维恩近似
def Wein(x, T, k, I0):
    a = 1.19e-16
    b = 1.44e-2
    l = x
    I = k*a/(l**5)*numpy.exp(-b/l/T)+I0
    return I

#线性函数
def Linear(x, a, b):
    return a * x + b

