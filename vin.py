#!/usr/bin/env python
#coding:utf-8
from keras.models import Model
from keras.layers import Input, Convolution2D, merge, Reshape, Dense, Lambda
import keras.backend as K

def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3

def vin_model(l_s=16, k=10, l_h=150, l_q=10, l_a=8):
    _handle_dim_ordering()
    def ext_start(inputs):
        m = inputs[0]
        s = inputs[1]
        w = K.one_hot(s[:, 0] + l_s * s[:, 1], l_s * l_s) # (None, l_s * l_s)
        return K.transpose(K.sum(w * K.permute_dimensions(m, (1, 0, 2)), axis=2))

    map_in = Input(shape=(l_s, l_s, 2) if K.image_dim_ordering() == 'tf' else (2, l_s, l_s))
    x = Convolution2D(l_h, 3, 3, subsample=(1, 1),
                      activation='relu',
                      border_mode='same')(map_in)
    r = Convolution2D(1, 1, 1, subsample=(1, 1),
                      border_mode='valid',
                      bias=False, name='reward')(x)
    conv3 = Convolution2D(l_q, 3, 3, subsample=(1, 1),
                          border_mode='same',
                          bias=False)
    conv3b = Convolution2D(l_q, 3, 3, subsample=(1, 1),
                           border_mode='same',
                           bias=False)
    q_ini = conv3(r)
    q = q_ini
    for idx in range(k):
        v = Lambda(lambda x: K.max(x, axis=CHANNEL_AXIS, keepdims=True),
                   output_shape=(l_s, l_s, 1) if K.image_dim_ordering() == 'tf' else (1, l_s, l_s),
                   name='value{}'.format(idx + 1))(q)
        q = merge([q_ini, conv3b(v)], mode='sum')

    if K.image_dim_ordering() == 'tf':
        q = Lambda(lambda x: K.permute_dimensions(x, (0, 3, 1, 2)), output_shape=(l_q, l_s, l_s))(q)
    q = Reshape(target_shape=(l_q, l_s * l_s))(q)
    s_in = Input(shape=(2,), dtype='int32')
    q_out = merge([q, s_in], mode=ext_start, output_shape=(l_q,))
    out = Dense(l_a, activation='softmax', bias=False)(q_out)
    return Model(input=[map_in, s_in], output=out)

def get_layer_output(model, layer_name, x):
    return K.function([model.layers[0].input], [model.get_layer(layer_name).output])([x])[0]

if __name__ == "__main__":
    from keras.utils import plot_model
    model = vin_model(k=20)
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
