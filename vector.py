import re
import os
import collections
import random
import tensorflow as tf
import numpy as np
import tqdm
import re
import math
import pandas as pd
from tensorflow.contrib.tensorboard.plugins import projector
from operator import add

#to find the folders in a directory
root, dirs, files = os.walk("/home/adil/NLP/hw_1/materials/code/dataset/DATA/TRAIN/").next()




for i in range (len(dirs)):

    Domain = dirs[i]

    print dirs[xia]
    #reading the number of files in each directory
    onlyfiles = next(os.walk('/home/adil/NLP/hw_1/materials/code/dataset/DATA/TRAIN/'+dirs[i]))[2]
    #no.of times loop should be repeated
    loop_len = len(onlyfiles)
    #removing stop words
    def stop_wrd(line):
        return ' '.join(''.join([i if ord(i) >=65 and ord(i) <=90 or  ord(i) >= 97 and ord(i) <= 122 else ' ' for i in line]).split())
    stoplist = set([w.rstrip('\r\n') for w in open("/home/adil/Desktop/test/stopwords.txt")])

    #opening the embeddings and saving it into a list
    with open('/home/adil/Desktop/NLP/3000001(3.4)/final_embedding_dic.txt') as my_file:
         testsite_array123 = my_file.readlines()
    testsite_array1234=np.array(testsite_array123)
    #opening the metadata and saving it into a list
    with open('/home/adil/Desktop/NLP/3000001(3.4)/metadata.tsv') as my_file:
        testsite_array1 = my_file.readlines()
        testsite_array12 = [elem.strip().split(';') for elem in testsite_array1]

    def read1(j):
        data=[]
        limit=20000
        with open('/home/adil/NLP/hw_1/materials/code/dataset/DATA/TRAIN/'+Domain+'/'+str(j+1)+'.txt') as file:

            for line in file.readlines():
                line = line.replace('+', ' ').replace('.', ' ').replace(',', ' ').replace(':', ' ')
                line=re.sub("(^|\W)\d+($|\W)", " ", line)
                line = get_latin(line)
                split = line.lower().strip().split()
                split = [word for word in split if (word not in stoplist) and (len(word)>1)]
                if limit > 0 and limit - len(split) < 0:
                    split = split[:limit]
                else:
                    limit -= len(split)
                if limit >= 0:

                    data += split
        dim= len(data)
        return dim,data

    for j in range(loop_len):

        dim,data=read1(j)
        #list containg metadata
        w=(testsite_array12)
        new_list = list()
        #make a new list of index of each words in the metadata which is in the new txt file
        for k in range((dim)):

            if [data[k]] in w:
                new_list.append(w.index([data[k]]))
        a=testsite_array123[1000]#for reference
        b=testsite_array123[0]#for reference
        d=(map(float, a.split(' ')))#for reference
        e=(map(float, b.split(' ')))#for reference

        vec=map(add, d, e)#for reference
        dia=np.zeros_like(vec)#for reference
        feature=np.zeros_like(vec)
        #adding all the embeddings corresponding to the words in the new list
        for l in range(len(new_list)):
            a=(map(float, testsite_array123[new_list[l]].split(' ')))
            features=(map(add, a, feature))

        dimn = len(features)

        feature_vector = map(lambda x: x/dimn, features)
        f=open('/home/adil/Desktop/data_pre/DEV/dataset_train.csv','ab')
	    f.write(str(feature_vector)+','+dirs[i]+'\n')


## So each file will be converted it into a embedding and at the end of it the domain name will be added so this can be easily fed into any ML algorithm
