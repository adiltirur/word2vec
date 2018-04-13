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

root, dirs, files = os.walk("/home/adil/NLP/hw_1/materials/code/dataset/DATA/TRAIN/").next()




for xia in range (len(dirs)):

    Domain = dirs[xia]
    #from gensim.parsing.porter import PorterStemmer
    limit=20000
    print dirs[xia]
    onlyfiles = next(os.walk('/home/adil/NLP/hw_1/materials/code/dataset/DATA/TRAIN/'+dirs[xia]))[2] #dir is your directory path as string
    loop_len = len(onlyfiles)

    def get_latin(line):
        return ' '.join(''.join([i if ord(i) >=65 and ord(i) <=90 or  ord(i) >= 97 and ord(i) <= 122 else ' ' for i in line]).split())
    stoplist = set([w.rstrip('\r\n') for w in open("/home/adil/Desktop/test/stopwords.txt")])


    with open('/home/adil/Desktop/NLP/3000001(3.4)/final_embedding_dic.txt') as my_file:
         testsite_array123 = my_file.readlines()
    testsite_array1234=np.array(testsite_array123)

    with open('/home/adil/Desktop/NLP/3000001(3.4)/metadata.tsv') as my_file:
        testsite_array1 = my_file.readlines()
        testsite_array12 = [elem.strip().split(';') for elem in testsite_array1]

    def read1(xi):
        data=[]
        limit=1000
        with open('/home/adil/NLP/hw_1/materials/code/dataset/DATA/TRAIN/'+Domain+'/'+str(xi+1)+'.txt') as file:

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
                    #len(data)
        asd= len(data)
        return asd,data

    for xi in range(loop_len):
        #print("fname",xi)

        da,data=read1(xi)
        #print da

     #   del data

        w=(testsite_array12)
        damn = list()
        for i in range((da)):

            if [data[i]] in w:
                damn.append(w.index([data[i]]))
        a=testsite_array123[1000]
        b=testsite_array123[0]
        d=(map(float, a.split(' ')))
        e=(map(float, b.split(' ')))

        vec=map(add, d, e)
        dia=np.zeros_like(vec)
        feature=np.zeros_like(vec)
        #print len(damn)
        for i in range(len(damn)):
            a=(map(float, testsite_array123[damn[i]].split(' ')))
            features=(map(add, a, feature))
                #print i
            #correct logic

        adil= (sum(features))
        #print adil
	#f=open('/home/adil/Desktop/data_pre/label.csv','ab')
	#f.write(dirs[xia]+'\n')
        #del data
        #print("#############################",xi+1)
        f=open('/home/adil/Desktop/data_pre/TEST/dataset.csv','ab')
	f.write(str(adil)+','+dirs[xia]+'\n')
                        #new_f
                    #np.savetxt('/home/adil/Desktop/NLP/NLP_HW01/test/'+domain+'.txt',feature_vec)



#df1= pd.read_csv('/home/adil/Desktop/data_pre/label.csv')
#df2= pd.read_csv('/home/adil/Desktop/data_pre/feature.csv')
#result = pd.concat([df2, df1], axis=1)
#result.to_csv('/home/adil/Desktop/data_pre/dataset.csv', sep='\t', encoding='utf-8', index = False)



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

