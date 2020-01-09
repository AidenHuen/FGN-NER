# encoding:utf-8
import pickle

import keras.backend as K
import numpy
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from preprocess import *
import ModelLib
import config
from test import F1
import preprocess
para = config.para
# para["data_pk_path"] = "./data/msr-seg.pk"
# para["fea_pk_path"] = "./data/msr-seg-fea.pk"
# para["dict_pk_path"] = "./data/msr-seg-dict.pk"
#
# para["train_path"] = config.fold_path + "MSRA/train.txt"
# para["test_path"] = config.fold_path + "MSRA/test.txt"

# para["data_pk_path"] = "./data/peo-pos.pk"
# para["fea_pk_path"] = "./data/peo-pos-fea.pk"
# para["dict_pk_path"] = "./data/peo-pos-dict.pk"
# para["train_path"] = config.fold_path + "peopleDaily/199801_train"
# para["test_path"] = config.fold_path + "peopleDaily/199801_val"

# para["data_pk_path"] = "./data/msra-ner.pk"
# para["fea_pk_path"] = "./data/msra-ner-fea.pk"
# para["dict_pk_path"] = "./data/msra-ner-dict.pk"
# para["train_path"] = config.fold_path + "MSRA/train_ner_turned.txt"
# para["data_pk_path"] = "./data/nlpcc-pos.pk"
# para["fea_pk_path"] = "./data/nlpcc-pos-fea.pk"
# para["dict_pk_path"] = "./data/nlpcc-pos-dict.pk"
# para["train_path"] = config.fold_path + "nlpcc2015/Pos_train.txt"
# para["test_path"] = config.fold_path + "nlpcc2015/Pos_test.txt"



x1_train, x2_train, y_train,  x1_test, x2_test, y_test, tags = pickle.load(open(para["data_pk_path"], "rb"))
word2id,radical2id, pinyin2id, rad2id, id2id_radical, id2id_pinyin, id2id_rad = pickle.load(open(para["dict_pk_path"], "rb"))
train_x, test_x, radical_train, radical_test, pinyin_train, pinyin_test, rad_train, rad_test, img_embed = pickle.load(open(para["fea_pk_path"], "rb"))
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
print(tags)
print(x1_train.shape,y_train.shape)
print(x1_test.shape,y_test.shape)
print(y_train)

def result_proess(pred_y, x1_test, tags):
    lengths = get_lengths(x1_test)
    tag_pred_y = []
    tag_val_y = []
    for i, y in enumerate(pred_y):
        y = [numpy.argmax(dim) for dim in y]
        # print(lengths[i])
        p_y = y[:lengths[i]]
        # print(p_y)
        v_y = y_test[i][:lengths[i]].flatten()
        # print(v_y)
        p_y = [tags[dim] for dim in p_y]
        v_y = [tags[dim] for dim in v_y]
        tag_pred_y.append(p_y)
        tag_val_y.append(v_y)
    return tag_pred_y,tag_val_y


def finetune_bert(para):
    para['tag_num'] = len(tags)
    para['rad_vocab_size'] = len(rad2id.keys()) + 1
    para['radical_vocab_size'] = len(radical2id.keys()) + 1
    para['pinyin_vocab_size'] = len(pinyin2id.keys()) + 1
    para["word_num"] = len(word2id.keys()) + 1
    para["fea_embed"] = None
    model = ModelLib.BERT(para)


    # checkpoint = ModelCheckpoint(para["model_path"], monitor='val_loss', verbose=1, #val_viterbi_acc
    #                              save_best_only=True, mode='min')
    # model.fit(x=[x1_train, x2_train], y=y_train, verbose=2, batch_size=para["batch_size"],
    #               callbacks=[checkpoint], validation_data=([x1_test,x2_test],y_test), epochs=para["EPOCHS"])
    F_max = 0
    for i in range(15):
        print("epochs:",i)
        model.fit(x=[x1_train, x2_train], y=y_train, verbose=1, batch_size=para["batch_size"], epochs=1)
        pred_y = model.predict([x1_test,x2_test],verbose=1,batch_size=64)
        tag_pred_y,tag_val_y = result_proess(pred_y, x1_test, tags)
        P, R, F = F1(tag_pred_y, tag_val_y)
        print("P:" + str(P))
        print("R:" + str(R))
        print("F1:" + str(F))
        if F> F_max:
            F_max = F
            model.save_weights(para["model_path"],overwrite=True)


def train_bert(para,feature="",use_bert=True,use_bert_embed=True):
    para['tag_num'] = len(tags)
    para['rad_vocab_size'] = len(rad2id.keys()) + 1
    para['radical_vocab_size'] = len(radical2id.keys()) + 1
    para['pinyin_vocab_size'] = len(pinyin2id.keys()) + 1
    para["word_num"] = len(word2id.keys()) + 1
    para["fea_embed"] = None

    bool_x = creat_bool_x(train_x)
    bool_x_test = creat_bool_x(test_x)

    if "img" in feature:
        para["img_embed_weight"] = img_embed

    if use_bert_embed == True:
        para["is_trainable"] = False
        model = ModelLib.FGN(para,feature=feature,use_bert=use_bert)
        bert = ModelLib.BERT(para)
        bert.load_weights(para["embed_path"])
        bert_layer = bert.get_layer("BERT")
        fea_embed = bert_layer.get_weights()
        try:
            model.get_layer("BERT").set_weights(fea_embed)
        except:
            pass
    else:
        para["is_trainable"] = True
        model = ModelLib.FGN(para,feature=feature,use_bert=use_bert)



    # checkpoint = ModelCheckpoint(para["model_path"], monitor='val_viterbi_acc', verbose=1, #val_viterbi_acc
    #                              save_best_only=True, mode='max')

    F_max = 0
    for i in range(para["EPOCHS"]):
        print("epochs:",i)
        if feature == "img":
            model.fit(x=[x1_train,x2_train,train_x,bool_x], y=y_train, epochs=1, verbose=1,
                      batch_size=para["batch_size"]) # validation_data=([x1_test,x2_test,test_x,bool_x_test],y_test)
            pred_y =  model.predict([x1_test,x2_test,test_x,bool_x_test])
        elif feature == "img&radical":
            model.fit(x=[x1_train,x2_train,train_x,bool_x,radical_train], y=y_train, epochs=1, verbose=1,batch_size=para["batch_size"],
                  ) # validation_data=([x1_test,x2_test,test_x,bool_x_test],y_test)
            pred_y =  model.predict([x1_test,x2_test,test_x,bool_x_test,radical_test])
        elif feature == "radical":
            model.fit(x=[x1_train,x2_train,radical_train], y=y_train, epochs=1, verbose=2,batch_size=para["batch_size"],
                  ) # validation_data=([x1_test,x2_test,test_x,bool_x_test],y_test)
            pred_y =  model.predict([x1_test,x2_test,radical_test])
        elif feature == "pinyin":
            model.fit(x=[x1_train, x2_train, pinyin_train], y=y_train, epochs=1, verbose=2,
                      batch_size=para["batch_size"],
                      )  # validation_data=([x1_test,x2_test,test_x,bool_x_test],y_test)
            pred_y = model.predict([x1_test, x2_test, pinyin_test])
        elif feature == "":
            model.fit(x=[x1_train,x2_train], y=y_train, epochs=1, verbose=2,
                      batch_size=para["batch_size"],
                      )  # validation_data=([x1_test,x2_test,test_x,bool_x_test],y_test)
            pred_y = model.predict([x1_test,x2_test])
        tag_pred_y,tag_val_y = result_proess(pred_y, x1_test, tags)
        try:
            P, R, F = F1(tag_pred_y, tag_val_y)
            # P, R, F = pos_F1(tag_pred_y, tag_val_y)
            print("P:" + str(P))
            print("R:" + str(R))
            print("F1:" + str(F))
            if F> F_max:
                F_max = F
                model.save_weights(para["model_path"],overwrite=True)
        except:
            pass

    # if feature == "img&radical":
    #     model.fit(x=[x1_train,x2_train,train_x,bool_x,radical_train],y=y_train, epochs=para["EPOCHS"], verbose=1,batch_size=para["batch_size"], callbacks=[checkpoint],
    #               validation_data=([x1_test,x2_test,test_x,bool_x_test,radical_test],y_test))

if __name__ == "__main__":

    para["char_dropout"] = 0.5
    para["rnn_dropout"] = 0.5
    dataset = "weibo-ner"


    # para["model_path"] = "./model/"+dataset+"/bert"
    # finetune_bert(para)
    para["embed_path"] = "./model/"+dataset+"/bert"
    para["model_path"] = "./model/"+dataset+"/bert-img" #&radical-outer
    train_bert(para,feature="img", use_bert=True,use_bert_embed=True)


    # para["model_path"] = "./model/"+dataset+"/bert-img&radical" #&radical-outer
    # train_bert(para,feature="img&radical",use_bert=True,use_bert_embed=True)
    # para["model_path"] = "./model/"+dataset+"/bert-img&radical-outer" #&radical-outer
    # train_bert(para,feature="img&radical",use_bert=True,use_bert_embed=True)
