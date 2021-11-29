import numpy as np
import keras.backend as K
from keras.callbacks import Callback
from keras.utils import to_categorical
from keras.layers import *
from keras.models import Model
from keras.models import Sequential
import keras.backend as K
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
from keras.metrics import categorical_accuracy
np.random.seed(1017)

class CNN:
    batch_size = 100
    name = ''
    dim = 2600
    classes = 3
    def identity_Block(self, inpt, nb_filter, kernel_size, strides=1, with_conv_shortcut=False):
        x = self.Conv1d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
        x = self.Conv1d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
        if with_conv_shortcut:
            shortcut = self.Conv1d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, inpt])
            return x

    def Conv1d_BN(self, x, nb_filter, kernel_size, strides=1, padding='same', name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv1D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = BatchNormalization(axis=1, name=bn_name)(x)
        return x

    def bottleneck_Block(self, inpt,nb_filters,strides=1,with_conv_shortcut=False):
        k1,k2,k3=nb_filters
        x = self.Conv1d_BN(inpt, nb_filter=k1, kernel_size=1, strides=strides, padding='same')
        x = self.Conv1d_BN(x, nb_filter=k2, kernel_size=3, padding='same')
        x = self.Conv1d_BN(x, nb_filter=k3, kernel_size=1, padding='same')
        if with_conv_shortcut:
            shortcut = self.Conv1d_BN(inpt, nb_filter=k3, strides=strides, kernel_size=1)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, inpt])
            return x

    def res18lm(self, name = None):
        input = Input(shape=(2607, 1))
        x = input[:,:2600]
        y = input[:,2604]
        x = ZeroPadding1D(3)(x)
        #conv1
        x = self.Conv1d_BN(x, nb_filter=32, kernel_size=10, strides=4, padding='valid')
        x = MaxPooling1D(pool_size=5, strides=3,padding='same')(x)
        #conv2_x
        x = self.identity_Block(x, nb_filter=32, kernel_size=3)
        x = self.identity_Block(x, nb_filter=32, kernel_size=3)
        #conv3_x
        x = self.identity_Block(x, nb_filter=64, kernel_size=3, strides=2, with_conv_shortcut=True)
        x = self.identity_Block(x, nb_filter=64, kernel_size=3)
        #conv4_x
        x = self.identity_Block(x, nb_filter=128, kernel_size=3, strides=2, with_conv_shortcut=True)
        x = self.identity_Block(x, nb_filter=128, kernel_size=3)
        #conv5_x
        x = self.identity_Block(x, nb_filter=128, kernel_size=3, strides=2, with_conv_shortcut=True)
        x = self.identity_Block(x, nb_filter=128, kernel_size=3)
        x = AveragePooling1D(pool_size=7)(x)
        x = Flatten()(x)
        y = Flatten()(y)
        x = concatenate([x, y])
        x = Dropout(0.2)(x)
        #x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(3, activation='softmax')(x)
        model = Model(inputs=input, outputs=x, name = name)
        return model

    def f1_loss(self, y_true, y_pred):
        loss = 0
        for i in np.eye(3):
            y_true_ = K.constant([list(i)]) * y_true
            y_pred_ = K.constant([list(i)]) * y_pred
            loss += 2.0 * K.sum(y_true_ * y_pred_) / K.sum(y_true_ + y_pred_ + K.epsilon())
        return -K.log(loss / 3.0 + K.epsilon())

    def score_metric(self, y_true, y_pred):
        y_true = K.argmax(y_true)
        y_pred = K.argmax(y_pred)
        score = 0.
        for i in range(3):
            y_true_ = K.cast(K.equal(y_true, i), 'float32')
            y_pred_ = K.cast(K.equal(y_pred, i), 'float32')
            score += 2.0 * K.sum(y_true_ * y_pred_) / K.sum(y_true_ + y_pred_ + K.epsilon())
        return score / 3.0

    def run(self, trainset, trainlabel, testset, testlabel, model, loss = 'categorical_crossentropy', rate = 1e-3, epochs = 10):
        trainlabel = to_categorical(np.array(trainlabel), 3)
        testlabel = to_categorical(np.array(testlabel), 3)
        model.summary()
        class Evaluate(Callback):
            def __init__(self):
                self.scores = []
                self.highest = 0.
            def on_epoch_end(self, epoch, logs=None):
                if logs['val_score_metric'] >= self.highest: # 保存最优模型权重
                    self.highest = logs['val_score_metric']
                    model.save_weights('models/{}/model'.format(model.name))
        evaluator = Evaluate()
        model.compile(loss=loss, optimizer=Adam(rate), 
                    metrics=[self.score_metric, categorical_accuracy])

        history = model.fit(trainset, trainlabel,
                            batch_size=self.batch_size,
                            epochs=epochs,
                            validation_data=(testset, testlabel),
                            callbacks=[evaluator])
        return model, history
    
    def __init__(self, batch_size = 128, name = None, dim = 2600):
        self.batch_size = batch_size
        self.name = name
        self.dim = dim