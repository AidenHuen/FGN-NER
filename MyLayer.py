# encoding: utf-8
import tensorflow as tf
from keras.engine import Layer
# from keras_self_attention import SeqSelfAttention,ScaledDotProductAttention
import math
import keras
from keras.layers import *
import numpy as np
from keras.callbacks import Callback
from keras.optimizers import Optimizer
from keras import backend as K, initializers, regularizers, constraints
from keras.engine.topology import Layer
from keras.layers import K, Activation
from keras.engine import Layer
import numpy as np
class Multiply(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Multiply, self).__init__(**kwargs)

    def call(self, x):
        return tf.multiply(x[0], x[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]

# repre = ImageEmbeding(img_weight = numpy.array)(x)
class ImageEmbeding(Layer):
    def __init__(self, output_dim,img_weight, **kwargs):
        self.output_dim = output_dim
        self.img_weight = img_weight
        super(ImageEmbeding, self).__init__(**kwargs)

    def call(self, x):
        return tf.nn.embedding_lookup(self.img_weight, x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],)+self.output_dim

class Split(Layer):

    def __init__(self,output_dim, **kwargs):
        self.output_dim = output_dim
        super(Split, self).__init__(**kwargs)

    def call(self, x):
        repre = tf.split(x,3,1)[1]
        return tf.reshape(repre, [-1,self.output_dim])

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[2])

from keras import backend as K
from keras.engine.topology import Layer

class Outer(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Outer, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None

    def build(self,input_shape):
        self.x_shape = input_shape
        self.built = True

    def call(self,x,mask=None):
        # print(x[0].shape[-1])
        if mask is not None:
            # print(mask)
            mask_1 = K.repeat(mask[0], x[0].shape[-1])
            mask_1 = tf.transpose(mask_1, [0,2,1])
            mask_1 = K.cast(mask_1, K.floatx())
            x1 = x[0] * mask_1
            mask_2 = K.repeat(mask[1], x[1].shape[-1])
            mask_2 = tf.transpose(mask_2, [0,2,1])
            mask_2 = K.cast(mask_2, K.floatx())
            x2 = x[1] * mask_2
            x1 = tf.reshape(x1,[-1,self.x_shape[0][1],1,self.x_shape[0][2]])
            x2 = tf.reshape(x2,[-1,self.x_shape[1][1],self.x_shape[1][2],1])
            # result = tf.add(x[0], x[1])
            result = tf.multiply(x1,x2)
            result = tf.reshape(result,[-1,self.x_shape[0][1],self.x_shape[1][2]*self.x_shape[0][2]])
            return(result)
        else:
            x1 = tf.reshape(x[0],[-1,self.x_shape[0][1],1,self.x_shape[0][2]])
            x2 = tf.reshape(x[1],[-1,self.x_shape[1][1],self.x_shape[1][2],1])
            # result = tf.add(x[0], x[1])
            result = tf.multiply(x1,x2)
            result = tf.reshape(result,[-1,self.x_shape[0][1],self.x_shape[1][2]*self.x_shape[0][2]])
            return result

    def compute_output_shape(self, input_shape):
    #     return (input_shape[0][0],input_shape[0][1],input_shape[1][2],input_shape[0][2],)
        return (input_shape[0][0], input_shape[0][1], input_shape[1][2]*input_shape[0][2])

class sliding_Outer(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(sliding_Outer, self).__init__(**kwargs)

    def compute_mask(self, input, mask=None):
        # need not to pass the mask to next layers

        return None

    def build(self,input_shape):
        self.x_shape = input_shape
        self.built = True

    def call(self,x,mask=None):
        # print(mask)
        if mask is not None:
            # print(x[0])
            embed_mask_1 = np.zeros(shape=(2,x[0].shape[-2],x[0].shape[-1]))
            embed_mask_1[1] = np.ones(shape=(x[0].shape[-2],x[0].shape[-1]))
            mask[0] = tf.cast(mask[0],dtype="int32")
            mask_1 = tf.nn.embedding_lookup(embed_mask_1,mask[0])
            mask_1 = tf.cast(mask_1,tf.float32)
            x1 = x[0] * mask_1
            embed_mask_2 = np.zeros(shape=(2,x[1].shape[-2],x[1].shape[-1]))
            embed_mask_2[1] = np.ones(shape=(x[1].shape[-2],x[1].shape[-1]))
            mask[1] = tf.cast(mask[1],dtype="int32")
            mask_2 = tf.nn.embedding_lookup(embed_mask_2, mask[1])
            mask_2 = tf.cast(mask_2,tf.float32)
            x2 = x[1] * mask_2

            x1 = tf.reshape(x1,[-1,self.x_shape[0][1],self.x_shape[0][2],1,self.x_shape[0][3]])
            x2 = tf.reshape(x2,[-1,self.x_shape[1][1],self.x_shape[0][2],self.x_shape[1][3],1])
            # result = tf.add(x[0], x[1])
            result = tf.multiply(x1,x2)
            result = tf.reshape(result,[-1,self.x_shape[0][1],self.x_shape[0][2],self.x_shape[1][3]*self.x_shape[0][3]])
            return result

    def compute_output_shape(self, input_shape):
    #     return (input_shape[0][0],input_shape[0][1],input_shape[1][2],input_shape[0][2],)
        return (input_shape[0][0],input_shape[0][1],input_shape[0][2],input_shape[0][3]*input_shape[1][3])

class MaskMeanPool(Layer):
    def __init__(self, axis, **kwargs):
        self.supports_masking = True
        self.axis = axis
        super(MaskMeanPool, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None

    def call(self, x, mask=None):
        if mask is not None:
            mask = K.repeat(mask, x.shape[-1])
            mask = tf.transpose(mask, [0,2,1])
            mask = K.cast(mask, K.floatx())
            x = x * mask
            # return K.mean(x, axis=self.axis)
            return K.sum(x, axis=self.axis) / K.sum(mask, axis=self.axis)
        else:
            return K.mean(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        output_shape = []
        for i in range(len(input_shape)):
            if i!=self.axis:
                output_shape.append(input_shape[i])
        return tuple(output_shape)


class MaskMaxPooling(Layer):
    def __init__(self, axis, **kwargs):
        self.supports_masking = True
        self.axis = axis
        super(MaskMaxPooling, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return input_mask

    def call(self, x, mask=None):
        if mask is not None:
            mask = K.repeat(mask, x.shape[-1])
            mask = tf.transpose(mask, [0, 2, 1])
            mask = K.cast(mask, K.floatx())
            x = x * mask
            # return K.sum(x, axis=self.axis) / K.sum(mask, axis=self.axis)
            return K.max(x, axis=self.axis)
        else:
            # return K.mean(x, axis=self.axis)
            return K.max(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        output_shape = []
        for i in range(len(input_shape)):
            if i != self.axis:
                output_shape.append(input_shape[i])
        return tuple(output_shape)


class PositionEmbedding(Layer):
    """定义位置Embedding，这里的Embedding是可训练的。
    """
    def __init__(self, input_dim, output_dim, merge_mode='add', **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode

    def compute_mask(self, input, mask=None):
        # need not to pass the mask to next layers
        # if input_mask is not None:
        #     return input.shape
        # else:
        #     return None
        return mask

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(name='embeddings',
                                          shape=(self.input_dim,
                                                 self.output_dim),
                                          initializer='RandomNormal')

    def call(self, inputs,mask=None):


        input_shape = K.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]
        pos_embeddings = self.embeddings[:seq_len]
        pos_embeddings = K.expand_dims(pos_embeddings, 0)
        pos_embeddings = K.tile(pos_embeddings, [batch_size, 1, 1])

        if mask is not None:
            mask_0 = K.repeat(mask, inputs.shape[-1])
            mask_0 = tf.transpose(mask_0, [0, 2, 1])
            mask_0 = K.cast(mask_0, K.floatx())
            inputs = inputs * mask_0
            mask_1 = K.repeat(mask, pos_embeddings.shape[-1])
            mask_1 = tf.transpose(mask_1, [0, 2, 1])
            mask_1 = K.cast(mask_1, K.floatx())
            pos_embeddings = pos_embeddings * mask_1

        if self.merge_mode == 'add':
            return inputs + pos_embeddings
        else:
            return K.concatenate([inputs, pos_embeddings])

    def compute_output_shape(self, input_shape):
        if self.merge_mode == 'add':
            return input_shape
        else:
            return input_shape[:2] + (input_shape[2] + self.v_dim, )

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class SeqSelfAttention(keras.layers.Layer):

    ATTENTION_TYPE_ADD = 'additive'
    ATTENTION_TYPE_MUL = 'multiplicative'

    def __init__(self,
                 units=32,
                 attention_width=None,
                 attention_type=ATTENTION_TYPE_ADD,
                 return_attention=False,
                 history_only=False,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_additive_bias=True,
                 use_attention_bias=True,
                 attention_activation=None,
                 attention_regularizer_weight=0.0,
                 **kwargs):
        """Layer initialization.

        For additive attention, see: https://arxiv.org/pdf/1806.01264.pdf

        :param units: The dimension of the vectors that used to calculate the attention weights.
        :param attention_width: The width of local attention.
        :param attention_type: 'additive' or 'multiplicative'.
        :param return_attention: Whether to return the attention weights for visualization.
        :param history_only: Only use historical pieces of data.
        :param kernel_initializer: The initializer for weight matrices.
        :param bias_initializer: The initializer for biases.
        :param kernel_regularizer: The regularization for weight matrices.
        :param bias_regularizer: The regularization for biases.
        :param kernel_constraint: The constraint for weight matrices.
        :param bias_constraint: The constraint for biases.
        :param use_additive_bias: Whether to use bias while calculating the relevance of inputs features
                                  in additive mode.
        :param use_attention_bias: Whether to use bias while calculating the weights of attention.
        :param attention_activation: The activation used for calculating the weights of attention.
        :param attention_regularizer_weight: The weights of attention regularizer.
        :param kwargs: Parameters for parent class.
        """
        super(SeqSelfAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.attention_width = attention_width
        self.attention_type = attention_type
        self.return_attention = return_attention
        self.history_only = history_only
        if history_only and attention_width is None:
            self.attention_width = int(1e9)

        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.attention_activation = keras.activations.get(attention_activation)
        self.attention_regularizer_weight = attention_regularizer_weight
        self._backend = keras.backend.backend()

        if attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self.Wx, self.Wt, self.bh = None, None, None
            self.Wa, self.ba = None, None
        elif attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self.Wa, self.ba = None, None
        else:
            raise NotImplementedError('No implementation for attention type : ' + attention_type)

    def get_config(self):
        config = {
            'units': self.units,
            'attention_width': self.attention_width,
            'attention_type': self.attention_type,
            'return_attention': self.return_attention,
            'history_only': self.history_only,
            'use_additive_bias': self.use_additive_bias,
            'use_attention_bias': self.use_attention_bias,
            'kernel_initializer': keras.regularizers.serialize(self.kernel_initializer),
            'bias_initializer': keras.regularizers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
            'attention_activation': keras.activations.serialize(self.attention_activation),
            'attention_regularizer_weight': self.attention_regularizer_weight,
        }
        base_config = super(SeqSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self._build_additive_attention(input_shape)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self._build_multiplicative_attention(input_shape)
        super(SeqSelfAttention, self).build(input_shape)

    def _build_additive_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wt = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wt'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wx = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wx'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_additive_bias:
            self.bh = self.add_weight(shape=(self.units,),
                                      name='{}_Add_bh'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)


        self.Wa = self.add_weight(shape=(self.units, 1),
                                  name='{}_Add_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Add_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def _build_multiplicative_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wa = self.add_weight(shape=(feature_dim, feature_dim),
                                  name='{}_Mul_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Mul_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def call(self, inputs, mask=None, **kwargs):
        input_len = K.shape(inputs)[1]
        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            e = self._call_additive_emission(inputs)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            e = self._call_multiplicative_emission(inputs)

        if self.attention_activation is not None:
            e = self.attention_activation(e)
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        if self.attention_width is not None:
            if self.history_only:
                lower = K.arange(0, input_len) - (self.attention_width - 1)
            else:
                lower = K.arange(0, input_len) - self.attention_width // 2
            lower = K.expand_dims(lower, axis=-1)
            upper = lower + self.attention_width
            indices = K.expand_dims(K.arange(0, input_len), axis=0)
            e = e * K.cast(lower <= indices, K.floatx()) * K.cast(indices < upper, K.floatx())
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask)
            e = K.permute_dimensions(K.permute_dimensions(e * mask, (0, 2, 1)) * mask, (0, 2, 1))
        # print(mask)
        # a_{t} = \text{softmax}(e_t)
        s = K.sum(e, axis=-1, keepdims=True)
        a = e / (s + K.epsilon())

        # l_t = \sum_{t'} a_{t, t'} x_{t'}
        v = K.batch_dot(a, inputs)
        if self.attention_regularizer_weight > 0.0:
            self.add_loss(self._attention_regularizer(a))

        if self.return_attention:
            return [v, a]
        return v

    def _call_additive_emission(self, inputs):
        input_shape = K.shape(inputs)
        batch_size, input_len = input_shape[0], input_shape[1]

        # h_{t, t'} = \tanh(x_t^T W_t + x_{t'}^T W_x + b_h)
        q = K.expand_dims(K.dot(inputs, self.Wt), 2)
        k = K.expand_dims(K.dot(inputs, self.Wx), 1)
        if self.use_additive_bias:
            h = K.tanh(q + k + self.bh)
        else:
            h = K.tanh(q + k)

        # e_{t, t'} = W_a h_{t, t'} + b_a
        if self.use_attention_bias:
            e = K.reshape(K.dot(h, self.Wa) + self.ba, (batch_size, input_len, input_len))
        else:
            e = K.reshape(K.dot(h, self.Wa), (batch_size, input_len, input_len))
        return e

    def _call_multiplicative_emission(self, inputs):
        # e_{t, t'} = x_t^T W_a x_{t'} + b_a
        e = K.batch_dot(K.dot(inputs, self.Wa), K.permute_dimensions(inputs, (0, 2, 1)))
        if self.use_attention_bias:
            e += self.ba[0]
        return e

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if self.return_attention:
            return [mask, None]
        return mask

    def _attention_regularizer(self, attention):
        batch_size = K.cast(K.shape(attention)[0], K.floatx())
        input_len = K.shape(attention)[-1]
        indices = K.expand_dims(K.arange(0, input_len), axis=0)
        diagonal = K.expand_dims(K.arange(0, input_len), axis=-1)
        eye = K.cast(K.equal(indices, diagonal), K.floatx())
        return self.attention_regularizer_weight * K.sum(K.square(K.batch_dot(
            attention,
            K.permute_dimensions(attention, (0, 2, 1))) - eye)) / batch_size

    @staticmethod
    def get_custom_objects():
        return {'SeqSelfAttention': SeqSelfAttention}

class LayerNormalization(Layer):
    """实现基本的Layer Norm，只保留核心运算部分
    """
    def __init__(self, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.epsilon = K.epsilon() * K.epsilon()

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)
        shape = (input_shape[-1], )
        self.gamma = self.add_weight(shape=shape,
                                     initializer='ones',
                                     name='gamma')
        self.beta = self.add_weight(shape=shape,
                                    initializer='zeros',
                                    name='beta')

    def compute_mask(self, input, mask=None):
        # need not to pass the mask to next layers
        # if input_mask is not None:
        #     return input.shape
        # else:
        #     return None
        return mask

    def call(self, inputs,mask=None):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        outputs *= self.gamma
        outputs += self.beta
        return outputs

class SlidingWindow(Layer):
    def __init__(self, window_size=8, stride=2, **kwargs):
        super(SlidingWindow, self).__init__(**kwargs)
        self.window_size = window_size
        self.stride = stride

    def compute_mask(self, inputs,mask=None):
        return mask

    def call(self, inputs, mask=False):
        print(mask)
        vec_len = inputs.shape[2]
        pad_num = (inputs.shape[2]-self.window_size)%self.stride
        pad_input = tf.pad(inputs,[[0,0],[0,0],[0,pad_num]])
        slice_tensor = []
        for i in range(0,vec_len+pad_num-self.window_size+1,self.stride):
            slice = tf.expand_dims(tf.slice(pad_input,[0,0,i],[-1,-1,self.window_size]),2)
            slice_tensor.append(slice)
        output_tensor = tf.concat(slice_tensor,axis=2)
        print(output_tensor.shape)
        return output_tensor

    def compute_output_shape(self, input_shape):
        shape = math.ceil((float(input_shape[2])-self.window_size)/self.stride)+1
        return input_shape[0], input_shape[1], int(shape), self.window_size


class WordAttention(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(WordAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.v = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_v'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(WordAttention, self).build(input_shape)


    def compute_mask(self, input, mask=None):
        return None

    def call(self, x, mask=None):
        print(mask)
        uit = K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1)


        if self.bias:
            uit += self.b
        # uit = K.tanh(uit)
        # uit = K.relu(uit)
        # uit = K.softmax(uit)
        uit = K.sigmoid(uit)
        ait = LayerNormalization()(K.squeeze(K.dot(uit, K.expand_dims(self.v)), axis=-1))
        a = K.exp(ait)
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

class GroupDense(Layer):
    """分组全连接
    输入输出跟普通Dense一样，但参数更少，速度更快。
    """
    def __init__(self,
                 units,
                 groups=2,
                 activation='linear',
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(GroupDense, self).__init__(**kwargs)
        self.units = units
        self.groups = groups
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)

    def compute_mask(self, input, mask=None):
        # need not to pass the mask to next layers
        # if input_mask is not None:
        #     return input.shape
        # else:
        #     return None
        return mask

    def build(self, input_shape):
        super(GroupDense, self).build(input_shape)
        input_dim = input_shape[-1]
        if not isinstance(input_dim, int):
            input_dim = input_dim.value
        assert input_dim % self.groups == 0
        assert self.units % self.groups == 0
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_dim // self.groups,
                                             self.units // self.groups,
                                             self.groups),
                                      initializer=self.kernel_initializer)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.units, ),
                                    initializer='zeros')

    def call(self, inputs, mask=None):
        ndim, shape = K.ndim(inputs), K.shape(inputs)
        shape = [shape[i] for i in range(ndim)]
        inputs = K.reshape(inputs, shape[:-1] + [shape[-1] // self.groups, self.groups])
        outputs = tf.einsum('...ig,ijg->...gj', inputs, self.kernel)
        outputs = K.reshape(outputs, shape[:-1] + [self.units])
        outputs = outputs + self.bias
        outputs = self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units, )

    def get_config(self):
        config = {
            'units': self.units,
            'groups': self.groups,
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
        }
        base_config = super(GroupDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

activations = keras.activations
class FeedForward(Layer):
    """FeedForward层，其实就是两个Dense层的叠加
    """
    def __init__(self,
                 units,
                 groups=1,
                 activation='relu',
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.units = units
        self.groups = groups
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        super(FeedForward, self).build(input_shape)
        output_dim = input_shape[-1]
        if not isinstance(output_dim, int):
            output_dim = output_dim.value
        if self.groups is None or self.groups == 1:
            self.dense_1 = Dense(units=self.units,
                                 activation=self.activation,
                                 kernel_initializer=self.kernel_initializer)
            self.dense_2 = Dense(units=output_dim,
                                 kernel_initializer=self.kernel_initializer)
        else:
            self.dense_1 = GroupDense(units=self.units,
                                      groups=self.groups,
                                      activation=self.activation,
                                      kernel_initializer=self.kernel_initializer)
            self.dense_2 = GroupDense(units=output_dim,
                                      groups=self.groups,

                                      kernel_initializer=self.kernel_initializer)
    def compute_mask(self, input, mask=None):
        # need not to pass the mask to next layers
        # if input_mask is not None:
        #     return input.shape
        # else:
        #     return None
        return mask

    def call(self, inputs, mask=None):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return x

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == "__main__":
    import tensorflow as tf
    import numpy as np
    from keras.layers import *
    from keras import Model
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # a = np.array([1,2,3])
    # b = np.array([2,3,4])
    # c = np.array([3,4,5])
    # a = np.outer(a, b)
    # a = np.outer(a,c)
    # print(a)
    # a = np.array([[[1, 2, 3, 4, 5],[1,2,3,0,0]],[[1, 2, 3, 4, 5],[1,2,3,0,0]]])
    # b = np.array([[[1, 2, 3, 4, 5],[1,2,3,0,0]],[[1, 2, 3, 4, 5],[1,2,3,0,0]]])
    #
    # from keras.initializers import ones
    # x_a = Input(shape=(2,5),)
    # x_n = Input(shape=(2,5),)
    #
    # x1 = TimeDistributed(Embedding(10, 10, mask_zero=True))(x_a)
    # x2 = TimeDistributed(Embedding(10, 10, mask_zero=True)(x_n))
    # # print(x1)
    # # x1 = TimeDistributed(PositionEmbedding(5,10))(x1)
    # # x1 = TimeDistributed(SeqSelfAttention(units=10,attention_activation="softmax"))(x1)
    # # result = TimeDistributed(LayerNormalization())(x1)
    #
    # outer = TimeDistributed(Outer())([x1,x2])
    #
    # model = Model(input=x_a,output=outer)
    # model.summary()
    # result = model.predict(a,batch_size=16)
    # print(result.shape)
    # for i in result:
    #     print(i)

    input = np.array([[1, 2, 0, 0, 0],[1,2,0,0,0]])
    x_a = Input(shape=(5,),dtype="int32")
    a = Embedding(10, 10, mask_zero=True)(x_a)
    b = Embedding(10,5,mask_zero=True)(x_a)
    a = SlidingWindow(window_size=6,stride=2)(a)
    b = SlidingWindow(window_size=3,stride=1)(b)
    a = sliding_Outer()([a,b])
    a = TimeDistributed(WordAttention())(a)
    model = Model(input=x_a,output=a)
    model.summary()
    result = model.predict(input)
    print(result.shape)
    print(result[1])
