import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
import os

# Building blocks
class BinaryClassifier(Layer):
    def __init__(self, activation = tf.nn.sigmoid):
        super(BinaryClassifier, self).__init__()
        self.activate = activation
        self.dense = layers.Dense(units = 1) 
        
    def call(self, input_tensor):
        return self.activate(self.dense(((input_tensor))))

class FeatureTokenizer(Layer):
    def __init__(self, numVar, d = 16):
        super(FeatureTokenizer, self).__init__()
        self.dense = layers.Dense(units = numVar * d)
        self.d = d
        self.reshape = layers.Reshape((numVar, d))
        
    def call(self, input_tensor):
        return self.reshape(self.dense(input_tensor))

class TransformerBlock(Layer):
    def __init__(self, num_heads = 10, keyValueDim = 16, mlpDropout = 0.1, atDropout = 0.2):
        super(TransformerBlock, self).__init__()
        
        # pre-norm and attention
        self.bn1 = layers.BatchNormalization()
        self.attention = layers.MultiHeadAttention(num_heads = num_heads,
                                                   key_dim = keyValueDim,
                                                   value_dim = keyValueDim,
                                                   dropout = atDropout)
        
        # pre-norm and MLP
        self.bn2 = layers.BatchNormalization()
        self.dense = layers.Dense(units = keyValueDim, activation = tf.nn.relu)
        self.drop = tf.keras.layers.Dropout(rate = mlpDropout)
        
        
    def call(self, input_tensor):  
        x = self.attention(self.bn1(input_tensor), self.bn1(input_tensor))
        x_inter = x + input_tensor
        x = self.drop(self.dense((self.bn2(x_inter))))
        return x + x_inter

# Model
class FT_Transformer3(Model):
    def __init__(self, numVar, d, num_heads, inp_shape):
        super(FT_Transformer3, self).__init__()
        self.ft = FeatureTokenizer(numVar = numVar, d = d)
        self.t1 = TransformerBlock(num_heads = num_heads, keyValueDim = d)
        self.t2 = TransformerBlock(num_heads = num_heads, keyValueDim = d)
        self.t3 = TransformerBlock(num_heads = num_heads, keyValueDim = d)
        self.max = layers.Maximum()
        self.final = BinaryClassifier()
        self.inp_shape = inp_shape
        
    def call(self, input_tensor):
        x = self.ft(input_tensor)
        x = self.t3(self.t2(self.t1(x)))
        x = tf.math.reduce_max(x, axis = 1)
        return self.final(x)
        
    def model(self):
        x = keras.Input(shape = self.inp_shape)
        return Model(inputs = x, outputs = self.call(x))

# Ensemble model
class FT_Transformer3_ensemble5(Model):
    def __init__(self, numVar, d, num_heads, inp_shape):
        super(FT_Transformer3_ensemble5, self).__init__()
        self.model1 = FT_Transformer3(numVar = numVar, d = d, num_heads = num_heads, inp_shape = inp_shape)
        self.model2 = FT_Transformer3(numVar = numVar, d = d, num_heads = num_heads, inp_shape = inp_shape)
        self.model3 = FT_Transformer3(numVar = numVar, d = d, num_heads = num_heads, inp_shape = inp_shape)
        self.model4 = FT_Transformer3(numVar = numVar, d = d, num_heads = num_heads, inp_shape = inp_shape)
        self.model5 = FT_Transformer3(numVar = numVar, d = d, num_heads = num_heads, inp_shape = inp_shape)
        self.inp_shape = inp_shape
        
    def call(self, input_tensor):
        x1 = self.model1(input_tensor)
        x2 = self.model2(input_tensor)
        x3 = self.model3(input_tensor)
        x4 = self.model4(input_tensor)
        x5 = self.model5(input_tensor)
        return (x1 + x2 + x3 + x4 + x5)/5
    
    def model(self):
        x = keras.Input(shape = self.inp_shape)
        return Model(inputs = x, outputs = self.call(x))
    

# Class
class FTT_model:
    def __init__(self, model_file):
        self.model = FT_Transformer3_ensemble5(numVar = 21, 
                                                d = 2, 
                                                num_heads = 2, 
                                                inp_shape = (50,))
        import os
        self.model.load_weights(f'{os.getcwd()}\{model_file}')
        self.model.compile()

    def build(model_file = 'FTT_weights_v1.tf'):
        return FTT_model(model_file)

    def predict(self, X):
        pred = self.model.predict(X)
        if pred[0] > 0.5:
            return 'Default'
        else: 
            return 'Not default'

