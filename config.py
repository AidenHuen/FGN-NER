fold_path = "/media/iiip/Seagate Backup Plus Drive/data/labelseq/"
para = {}

# data_name = "ontonote4"
# data_name = "resume"
data_name = "weibo"
# data_name = "ud1"
# data_name = "msra"
# data_name = "ctb5"
# data_name = "cityu"
task = "ner"
para["data_pk_path"] = "./data/"+data_name+".pk"
para["fea_pk_path"] = "./data/"+data_name+"-fea.pk"
para["dict_pk_path"] = "./data/"+data_name+"-dict.pk"



para["train_path"] = fold_path +task +"/"+data_name+"/train.char.bmes"
para["test_path"]=fold_path + task + "/"+data_name+"/test.char.bmes"

para["model_path"] = "./model/pku/lstm-crf-radical-embed-bert"
para["embed_path"] = fold_path + "50_vectors.txt"
para["traditional_dict_path"] = fold_path+"dict.txt"
para['bert_path'] = '/media/iiip/Seagate Backup Plus Drive/data/chinese_L-12_H-768_A-12/' #goolge pre-trained BERT
# para['bert_path'] = '/media/iiip/Seagate Backup Plus Drive/data/bert_wwm/

para["img_w"] = 50
para["img_h"] = 50
para["embed_dim"] = 400
para["unit_num"] = 200
para["split_seed"] = 2018
# para["max_len"] = 85
para["max_len"] = 100
para["EPOCHS"] = 30
para["batch_size"] = 16

para["traditional_chinese"] = False
para["sep"] = " "
para["char_dropout"] = 0.5
para["rnn_dropout"] = 0.5
para["lstm_unit"] = 768
para["REPRE_NUM"] = 64
para["fea_dropout"] = 0.2
para["fea_lstm_unit"] = 64
para["fea_dim"] = 32
para["radical_max"] = 7
para["pinyin_max"] = 8
para["lr"] = 0.002
# para["lr"] = 0.01
para["bert_lr"] = 0.00002
# para["bert_lr"] = 0.00001
embed_path = fold_path+""
traditional_dict_path = fold_path+"dict.txt"


