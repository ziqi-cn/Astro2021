import numpy
import pandas
import feather
import csv
import json
import matplotlib.pyplot as plt
from .log import *

config_file = "config/config.json"
Config = {}

def load_config(config_file=config_file):
    f = open(config_file)
    Config.update(json.load(f))
    logger.setLevel(log_level(Config["loglevel"]))
    logger.info("Log level: {}".format(Config["loglevel"]))

def safe_float(s):
    try:
        return float(s)
    except:
        return s

def load_data():
    data = []
    id = []
    for file in Config["datasets"]:
        logger.info("Loading {}".format(file))
        csvfile = open(file)
        reader = csv.reader(csvfile)
        rows = [list(map(safe_float, row)) for row in reader]
        rows = rows[1:]
        data += [row[:-1] for row in rows]
        id += [row[-1] for row in rows]
    return numpy.array(data), id

def label_to_int(label):
    if label == "star":
        return 0
    if label == "galaxy":
        return 1
    if label == "qso":
        return 2
    return label

def load_label(Files, id):
    p = 0
    label = []
    for file in Files:
        logger.info("Loading {}".format(file))
        csvfile = open(file)
        reader = csv.reader(csvfile)
        rows = [list(map(label_to_int, row)) for row in reader]
        rows = rows[1:]
        label += [-1] * len(rows)
        for row in rows:
            if id[p] == row[0]:
                label[p] = row[1]
                p += 1
            else:
                for i in range(len(id)):
                    if id[i] == row[0]:
                        label[i] = row[1]
                        p = i + 1
    
    return label

def load_feather(filepath):
    data = pandas.read_feather(filepath)
    data = data.to_numpy()
    return data[:,:-1], data[:, -1].astype(numpy.int8)

def save_(model, history, name):
    #model.save_weights('models/{}/model'.format(name))
    epochs=range(len(history.history['score_metric']))
    plt.figure()
    plt.plot(epochs,history.history['score_metric'],'b',label='Training score metric')
    plt.plot(epochs,history.history['val_score_metric'],'r',label='Validation score metric')
    plt.title('Traing and Validation score metric')
    plt.legend()
    plt.savefig('models/{}_score.jpg'.format(name))
    plt.figure()
    plt.plot(epochs,history.history['loss'],'b',label='Training loss')
    plt.plot(epochs,history.history['val_loss'],'r',label='Validation val_loss')
    plt.title('Traing and Validation loss')
    plt.legend()
    plt.savefig('models/{}_loss.jpg'.format(name))