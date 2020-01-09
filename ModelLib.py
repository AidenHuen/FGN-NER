import keras
import numpy
from keras import Model
from keras.layers import Embedding, Bidirectional, LSTM, \
    BatchNormalization, Dropout, Reshape, Conv2D, \
    Masking, MaxPooling2D, MaxPooling1D,Conv3D
from keras.layers import Input, Dense, Concatenate, \
    TimeDistributed,Permute,RepeatVector, Multiply, GRU,CuDNNGRU,GlobalMaxPool1D,Add
from keras_bert import load_trained_model_from_checkpoint
from keras_contrib.layers import CRF
from preprocess import *
import MyLayer
from keras.layers.core import *


SINGLE_ATTENTION_VECTOR = False


def add_img_repre(bool_input,para,char_input):
    # bool_input = Input(shape=(para["max_len"], para["REPRE_NUM"]), name='bool_input')
    # bool_input = Input(shape=(para["max_len"]+2,), name='bool_input')
    embed = MyLayer.ImageEmbeding(img_weight=para["img_embed_weight"],
                                  output_dim=(para["max_len"]+2, para["img_h"], para["img_w"], 1))(char_input)
    drop = Dropout(0.2)(embed)
    drop = Conv3D(filters=4,kernel_size=(3,3,3),strides=(1,1,1),padding="same",data_format="channels_last")(drop) #,activation="relu",
    drop = Conv3D(filters=8,kernel_size=(3,3,3),strides=(1,1,1),padding="same",data_format="channels_last")(drop)
    conv = TimeDistributed(Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), data_format="channels_last"))(drop)
    conv = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format="channels_last"))(conv)
    conv = TimeDistributed(Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), data_format="channels_last"))(conv)
    pool = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format="channels_last"))(conv)
    conv = TimeDistributed(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), data_format="channels_last"))(pool)
    pool = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format="channels_last"))(conv)
    conv = TimeDistributed(Conv2D(filters=para["REPRE_NUM"], kernel_size=(2, 2), strides=(1, 1), data_format="channels_last"))(pool)

    pool = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format="channels_last"))(conv)
    # pool = TimeDistributed(
    #     Conv2D(filters=para["REPRE_NUM"], kernel_size=(2, 2), strides=(1, 1), data_format="channels_last"))(pool)

    pool = Reshape((para["max_len"]+2,4,para["REPRE_NUM"]))(pool)
    fea_repre = MyLayer.MaskMaxPooling(axis=2)(pool)
    # fea_repre = MyLayer.MaskMeanPool(axis=2)(pool)
    # para["REPRE_NUM"] = para["REPRE_NUM"] * 4
    # fea_repre = Reshape((para["max_len"]+2,para["REPRE_NUM"]))(pool)
    bool_embed = Embedding(input_dim=2, output_dim=para["REPRE_NUM"],
                           input_length=para["max_len"]+2,trainable=False,weights=[creat_bool_embed(para["REPRE_NUM"])])(bool_input)
    fea_repre = MyLayer.Multiply(output_dim=(para["max_len"]+2, para["REPRE_NUM"]))([fea_repre, bool_embed])
    fea_repre = Masking()(fea_repre)
    # fea_repre = BatchNormalization()(fea_repre)

    return fea_repre


def add_fea_repre(bool_input,para, max_char_size, char_vocab_size):
    length = max_char_size *(para["max_len"]+2)
    fea_input = Input(shape=(length,), dtype='int32')

    fea_emb = Embedding(char_vocab_size, para["fea_dim"], input_length=length,trainable=False)(fea_input)
    fea_emb = Dropout(para["fea_dropout"])(fea_emb)
    fea_emb = Reshape((para["max_len"]+2, max_char_size,-1))(fea_emb)
    # fea_repre = TimeDistributed(Bidirectional(LSTM(para["fea_dim"], return_sequences=True),merge_mode="sum"))(fea_emb)
    # fea_repre = TimeDistributed(MyLayer.PositionEmbedding(para["max_len"]+2,para["fea_dim"]))(fea_emb)
    fea_repre = TimeDistributed(MyLayer.SeqSelfAttention(units=para["fea_dim"],attention_activation="tanh"))(fea_emb)
    # fea_repre = MyLayer.LayerNormalization()(Add()([fea_repre,fea_emb]))
    fea_repre = MyLayer.MaskMaxPooling(axis=2)(fea_repre)
    # bool_embed = Embedding(input_dim=2,output_dim=para["fea_dim"],
    #                        input_length=para["max_len"]+2,trainable=False,weights=[creat_bool_embed(para["fea_dim"])])(bool_input)
    # fea_repre = MyLayer.Multiply(output_dim=(para["max_len"]+2, para["fea_dim"]))([fea_repre, bool_embed])
    fea_repre = Masking()(fea_repre)
    return fea_input, fea_repre


def BERT(para):
    x1 = Input(shape=(para["max_len"]+2,))
    x2 = Input(shape=(para["max_len"]+2,))

    config_path = para["bert_path"] + 'bert_config.json'
    checkpoint_path = para["bert_path"] + "bert_model.ckpt"

    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)

    bert_model.name = "BERT"

    for l in bert_model.layers:
        l.trainable = True
    x = bert_model([x1, x2])

    crf = CRF(para["tag_num"], sparse_target=True)
    crf_output = crf(x)
    model = Model(input=[x1,x2],output = crf_output)
    model.summary()
    adam_0 = keras.optimizers.Adam(lr= para["bert_lr"], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(adam_0, loss=crf.loss_function, metrics=[crf.accuracy])
    return model


def FGN(para, feature="", use_bert=True):
    # for key in para:
    #     print key,para[key]
    x1 = Input(shape=(para["max_len"]+2,))
    x2 = Input(shape=(para["max_len"]+2,))

    if use_bert == True:
        config_path = para["bert_path"] + 'bert_config.json'
        checkpoint_path = para["bert_path"] + "bert_model.ckpt"
        bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
        for l in bert_model.layers:
            l.trainable = para["is_trainable"]
        bert_model.name = "BERT"
        repre = bert_model([x1, x2])
        # repre = TimeDistributed(Dense(units=768,activation="tanh"))(repre)
        # repre = BatchNormalization()(repre)
    else:
        repre = Embedding(para["word_num"], para["embed_dim"], trainable=False,mask_zero=True)(x1)
        repre = Dropout(para["char_dropout"])(repre)
    # repre = Dense(300, activation="relu")(repre)
    if feature == "img&radical":
        char_input = Input(shape=(para["max_len"] + 2,), dtype='int32', name='char_input')
        bool_input = Input(shape=(para["max_len"] + 2,), name='bool_input')
        img_repre = add_img_repre(bool_input, para, char_input)
        # repre = BatchNormalization()(repre)
        # img_repre = BatchNormalization()(img_repre)
        slice_repre = MyLayer.SlidingWindow(window_size=96, stride=12)(repre)
        slice_img = MyLayer.SlidingWindow(window_size=8, stride=1)(img_repre)
        fusion_repre = MyLayer.sliding_Outer()([slice_repre, slice_img])
        fusion_repre = TimeDistributed(MyLayer.WordAttention())(fusion_repre)
        fusion_repre = Dropout(0.2)(fusion_repre)
        fea_input, radical_repre = add_fea_repre(bool_input,
                                                 para, para['radical_max'], para['radical_vocab_size'])
        # slice_fusion = MyLayer.SlidingWindow(window_size=48, stride=24)(repre)
        # slice_radical = MyLayer.SlidingWindow(window_size=2, stride=1)(radical_repre)
        # fusion_repre_1 = MyLayer.sliding_Outer()([slice_fusion, slice_radical])
        # fusion_repre_1 = TimeDistributed(MyLayer.WordAttention())(fusion_repre_1)
        # # bool_embed_1 = Embedding(input_dim=2, output_dim=96, input_length=para["max_len"] + 2,
        # #                        weights=[creat_bool_embed(96)])(bool_input)
        # # fusion_repre_1 = Multiply()([fusion_repre_1, bool_embed_1])
        # fusion_repre_1 = Masking()(fusion_repre_1)
        repre = Concatenate()([repre,fusion_repre,img_repre,radical_repre])



    if feature == "img":
        char_input = Input(shape=(para["max_len"]+2,), dtype='int32', name='char_input')
        bool_input = Input(shape=(para["max_len"] + 2,), name='bool_input')
        img_repre = add_img_repre(bool_input, para, char_input)
        # repre = BatchNormalization()(repre)

        slice_repre = MyLayer.SlidingWindow(window_size=96, stride=12)(repre)
        slice_img = MyLayer.SlidingWindow(window_size=8, stride=1)(img_repre)
        fusion_repre = MyLayer.sliding_Outer()([slice_repre, slice_img])
        fusion_repre = TimeDistributed(MyLayer.WordAttention())(fusion_repre)
        # fusion_repre = MyLayer.Outer()([BatchNormalization()(repre),BatchNormalization()(img_repre)])
        # fusion_repre = TimeDistributed(Dense(128,activation="tanh"))(fusion_repre)
        # fusion_repre = Dropout(0.2)(fusion_repre)
        # repre = Add()([fusion_repre,repre])
        repre = Concatenate(axis=2)([repre,img_repre,fusion_repre])
        # repre = fusion_repre
        # repre = TimeDistributed(Dense(units=768,activation="relu"))(repre)

    elif feature == "radical":

        fea_input, radical_repre = add_fea_repre(bool_input,
                                                 para, para['radical_max'], para['radical_vocab_size'])
        slice_fusion = MyLayer.SlidingWindow(window_size=48, stride=24)(repre)
        slice_radical = MyLayer.SlidingWindow(window_size=2, stride=1)(radical_repre)
        fusion_repre_1 = MyLayer.sliding_Outer()([slice_fusion, slice_radical])
        fusion_repre_1 = TimeDistributed(MyLayer.WordAttention())(fusion_repre_1)
        repre = Concatenate()([repre, fusion_repre_1, radical_repre])

    elif feature == "pinyin":
        fea_input, pinyin_repre = add_fea_repre(bool_input,
                                                 para, para['pinyin_max'], para['pinyin_vocab_size'])
        slice_fusion = MyLayer.SlidingWindow(window_size=48, stride=24)(repre)
        slice_radical = MyLayer.SlidingWindow(window_size=2, stride=1)(radical_repre)
        fusion_repre_1 = MyLayer.sliding_Outer()([slice_fusion, slice_radical])
        fusion_repre_1 = TimeDistributed(MyLayer.WordAttention())(fusion_repre_1)

        repre = Concatenate()([repre, fusion_repre_1, pinyin_repre])

    # transformer
    # repre = MyLayer.PositionEmbedding(para["max_len"]+2,repre.shape.as_list()[2])(repre)
    # att_repre_list = []
    # att_repre_list.append(MyLayer.SeqSelfAttention(units=400,attention_activation="softmax")(repre))
    # att_repre_list.append(MyLayer.SeqSelfAttention(units=400,attention_activation="softmax")(repre))
    # att_repre_list.append(MyLayer.SeqSelfAttention(units=400,attention_activation="softmax")(repre))
    # att_repre_list.append(MyLayer.SeqSelfAttention(units=400,attention_activation="softmax")(repre))
    # repre = Add()(att_repre_list)
    # att_repre = Dropout(0.3)(repre)
    # repre = MyLayer.LayerNormalization()(att_repre)
    # feed_repre = MyLayer.FeedForward(units=1600)(repre)
    # feed_repre = Dropout(0.3)(feed_repre)
    # repre = Add()([repre,feed_repre])
    # repre = MyLayer.LayerNormalization()(repre)


    # repre = LSTM(para["lstm_unit"], return_sequences=True, dropout= para["rnn_dropout"])(repre)
    # repre = Bidirectional(LSTM(para["lstm_unit"], return_sequences=True, dropout=para["rnn_dropout"]),merge_mode="sum")(repre)

    crf = CRF(para["tag_num"], sparse_target=True)
    crf_output = crf(repre)

    if feature == "img&radical":
        model = Model(input=[x1,x2, char_input, bool_input, fea_input], output=crf_output)
    if feature == "img":
        model = Model(input=[x1,x2, char_input,bool_input],output=crf_output)
    elif feature == "radical" or feature =="pinyin":
        model = Model(input=[x1,x2, fea_input],output=crf_output)
    elif feature == "":
        model = Model(input=[x1,x2], output=crf_output)
    model.summary()
    adam_0 = keras.optimizers.Adam(lr=para["lr"], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(adam_0, loss=crf.loss_function, metrics=[crf.accuracy])
    return model
