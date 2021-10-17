import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer


# Building blocks
class BinaryClassifier(Layer):
    def __init__(self, activation = tf.nn.sigmoid):
        super(BinaryClassifier, self).__init__()
        self.activate = activation
        self.dense = layers.Dense(units = 1) 
        
    def call(self, input_tensor):
        return self.activate(self.dense(((input_tensor))))
    
class ResNetBlock(Layer):
    def __init__(self, hiddenUnits = 16, resDropoutRate = 0.25, hiddenDropoutRate = 0.25, activation = tf.nn.relu):
        super(ResNetBlock, self).__init__()
        self.bn = layers.BatchNormalization()
        self.hiddenDrop = layers.Dropout(hiddenDropoutRate)
        self.dense1 = layers.Dense(units = hiddenUnits, activation = activation)
        self.dense2 = layers.Dense(units = hiddenUnits)
        self.resDrop = layers.Dropout(resDropoutRate)
    
    def call(self, input_tensor):
        return input_tensor + self.resDrop(self.dense2(self.hiddenDrop(self.dense1(self.bn(input_tensor)))))

# Model
class ResNet2(Model):
    def __init__(self, hiddenUnits, inp_shape):
        super(ResNet2, self).__init__()
        self.inp_shape = inp_shape
        self.dense = layers.Dense(units = hiddenUnits)
        self.res1 = ResNetBlock(hiddenUnits)
        self.res2 = ResNetBlock(hiddenUnits)

        self.final = BinaryClassifier()
        
    def call(self, input_tensor):
        x = self.dense(input_tensor)
        x = self.res2(self.res1(x))
        return self.final(x)
    
    def model(self):
        x = keras.Input(shape = self.inp_shape)
        return Model(inputs = x, outputs = self.call(x))

# Ensemble model
class ResNet2_ensemble5(Model):
    def __init__(self, hiddenUnits, inp_shape):
        super(ResNet2_ensemble5, self).__init__()
        self.inp_shape = inp_shape
        self.res1 = ResNet2(hiddenUnits, inp_shape)
        self.res2 = ResNet2(hiddenUnits, inp_shape)
        self.res3 = ResNet2(hiddenUnits, inp_shape)
        self.res4 = ResNet2(hiddenUnits, inp_shape)
        self.res5 = ResNet2(hiddenUnits, inp_shape)
        
        
    def call(self, input_tensor):
        x1 = self.res1(input_tensor)
        x2 = self.res2(input_tensor)
        x3 = self.res3(input_tensor)
        x4 = self.res4(input_tensor)
        x5 = self.res5(input_tensor)
        x = (x1 + x2 + x3 + x4 + x5)/5
        return x
    
    def model(self):
        x = keras.Input(shape = self.inp_shape)
        return Model(inputs = x, outputs = self.call(x))

# Class 
class Resnet_model:
    
    def __init__(self, model_file):
        import os
        self.model = ResNet2_ensemble5(2, (50,))
        self.model.load_weights(f'{os.getcwd()}\{model_file}')
        self.model.compile()

    def build(model_file = 'resnet_weights_v1.tf'):
        return Resnet_model(model_file)

    def predict(self, X):
        pred = self.model.predict(X)
        if pred[0] > 0.5:
            return 'Default'
        else: 
            return 'Not default'

