
import numpy
from preprocess import *
from preprocess import get_lengths
import ModelLib
import config
import pickle
import datetime

para = config.para
# para["data_pk_path"] = "./data/msr-seg.pk"2

# para["data_pk_path"] = "./data/peo-pos.pk"
# para["fea_pk_path"] = "./data/peo-pos-fea.pk"
# para["dict_pk_path"] = "./data/peo-pos-dict.pk"
# para["train_path"] = config.fold_path + "peopleDaily/199801_train"
# para["test_path"] = config.fold_path + "peopleDaily/199801_val"

# para["data_pk_path"] = "./data/msra-ner.pk"
# para["fea_pk_path"] = "./data/msra-ner-fea.pk"
# para["dict_pk_path"] = "./data/msra-ner-dict.pk"
# para["train_path"] = config.fold_path + "MSRA/train_ner_turned.txt"
# para["test_path"] = config.fold_path + "MSRA/test_ner_turned.txt"

# para["data_pk_path"] = "./data/nlpcc-pos.pk"
# para["fea_pk_path"] = "./data/nlpcc-pos-fea.pk"
# para["dict_pk_path"] = "./data/nlpcc-pos-dict.pk"
# para["train_path"] = config.fold_path + "nlpcc2015/Pos_train.txt"
# para["test_path"] = config.fold_path + "nlpcc2015/Pos_test.txt"


x1_train, x2_train, y_train, x1_test, x2_test, y_test, tags = pickle.load(open(para["data_pk_path"], "rb"))
word2id, radical2id, pinyin2id, rad2id, id2id_radical, id2id_pinyin, id2id_rad = pickle.load(
    open(para["dict_pk_path"], "rb"))
train_x, test_x, radical_train, radical_test, pinyin_train, pinyin_test, rad_train, rad_test, img_embed = pickle.load(
    open(para["fea_pk_path"], "rb"))




def predict_BERT(para,feature="radical", use_bert=True):
    para['tag_num'] = len(tags)
    para['rad_vocab_size'] = len(rad2id.keys()) + 1
    para['radical_vocab_size'] = len(radical2id.keys()) + 1
    para['pinyin_vocab_size'] = len(pinyin2id.keys()) + 1
    para["word_num"] = len(word2id.keys()) + 1
    para["fea_embed"] = None


    bool_x_test = creat_bool_x(test_x)

    if "img" in feature:
        para["img_embed_weight"] = img_embed
    para["is_trainable"] = False
    model = ModelLib.FGN(para,feature=feature,use_bert=use_bert)


    # else:
    #     model = ModelLib.NORMAL_MODEL(para,feature=feature)
    model.load_weights(filepath=para["model_path"])

    if feature == "":
        pred_y = model.predict([x1_test,x2_test],batch_size=64,verbose=1)
    elif feature == "radical":
        pred_y = model.predict([x1_test,x2_test,radical_train],batch_size=64,verbose=1)
    elif feature == "pinyin":
        pred_y = model.predict([x1_test, x2_test, pinyin_train], batch_size=64, verbose=1)
    elif feature == "img":
        pred_y = model.predict([x1_test, x2_test,test_x,bool_x_test],batch_size=64, verbose=1)
    elif feature == "img&radical":
        pred_y  =model.predict([x1_test,x2_test,test_x,bool_x_test,radical_test],batch_size=64,verbose=1)
    lengths = get_lengths(x1_test)
    tag_pred_y = []
    tag_val_y = []
    for i, y in enumerate(pred_y):
        y = [numpy.argmax(dim) for dim in y]
        print(lengths[i])
        p_y = y[:lengths[i]]
        print(p_y)
        v_y = y_test[i][:lengths[i]].flatten()
        print(v_y)
        p_y = [tags[dim] for dim in p_y]
        v_y = [tags[dim] for dim in v_y]
        tag_pred_y.append(p_y)
        tag_val_y.append(v_y)
    return tag_pred_y,tag_val_y


def char_seg_acc(tag_pred_y, tag_val_y):


    acc = 0.0
    num = 0.0
    for j in range(len(tag_pred_y)):
        for z in range(len(tag_pred_y[j])):
            if tag_pred_y[j][z] == tag_val_y[j][z]:
                acc+=1
            num += 1
    print("test acc:"+str(acc/num))

def F1(y_pred,y):
    c = 0
    true = 0
    pos = 0
    for i in xrange(len(y)):
        start = 0
        for j in xrange(len(y[i])):
            if y_pred[i][j][0] == 'E' or y_pred[i][j][0] == 'S':
                pos += 1
            if y[i][j][0] == 'E' or y[i][j][0] == 'S':
                flag = True
                if y_pred[i][j] != y[i][j]:
                    flag = False
                if flag:
                    if y[i][j][0] == "E":
                        indexs = range(j)
                        indexs.reverse()
                        # print(indexs)
                        for k in indexs:
                            if y[i][k] != y_pred[i][k]:
                                flag = False
                            if y[i][k][0] == "B":
                                start = k
                                if y[i][start] != y_pred[i][start]:
                                    flag = False
                                break

                        if flag == True:
                            c += 1
                    if y[i][j][0] == "S":
                        c += 1
                true += 1
                # start = j+1

    P = c/float(pos)
    R = c/float(true)
    F = 2*P*R/(P+R)
    return P,R,F



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

if __name__ == "__main__":
    para["char_dropout"] = 0.5
    para["rnn_dropout"] = 0.5
    dataset = "onto-ner"
    para["model_path"] = "./model/" + dataset + "/bert-img"  # &radical-outer
    # pred_y, val_y = predict_bert(para, feature="img&radical")

    pred_y, val_y = predict_BERT(para,feature="img",use_bert=True)
    P,R,F = F1(pred_y,val_y)

    print("P:"+str(P))
    print("R:"+str(R))
    print("F1:"+str(F))
