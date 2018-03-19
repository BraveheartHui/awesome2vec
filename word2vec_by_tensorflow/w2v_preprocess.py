# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 22:00:45 2018

tensorflow版word2vec(skip-gram模型)实现:
    对分好词的语料进行处理，包括
        1 read_data_to_sentences 语料转换为句子序列
        2 build_dataset 获得单词表、单词index构成的语料
        3 divide_sentences_by_batch 将句子按照batch_words进行划分

python: 3.6.2
tensorflow:1.4.0

@author: hxh
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections




def read_data_to_sentences(filename):
    "convert data to a list of sentences"
    sentences_list = list()
    raw_words = list()
    with open(filename, encoding='UTF-8') as f:
        for sentence in f.readlines():
            sen = sentence.split()
            sentences_list.append(sen)
            for w in sen:
                raw_words.append(w)
    return  sentences_list,raw_words






# -------------------------------------------------------------------

# Step 2: 构建词汇表，统计每个单词出现的次数，定义最小词频，去掉除最小词频之外的词


def build_dataset(words,sentences,min_count):
    '''Process raw inputs into a dataset.'''
    count = list()
    # extend用于在末尾追加另一个list的多个值
    # 按照词序降序排列
    dic = sorted(collections.Counter(words).items(), key = lambda k: k[1],reverse=True) 
    # 返回词数大于min_count的单词
    count.extend((k,v) for k, v in dic if v >= min_count)
    dictionary = dict()
    
    # 当前count中保存词频大于等于min_count的词及其对应的词频
    for word, _ in count:
        # 保存每个词的序号
        dictionary[word] = len(dictionary)
    data = list()
    index = -1
    # sentences中保存全部单词
    for sentence in sentences:
        #print(sentence)
        sen_data = list()
        for word in sentence:
            if word in dictionary:
                index = dictionary[word]
                #data list 以序号的形式保存文本
                sen_data.append(index)
        #print(sen_data)
        data.append(sen_data)
    
    # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组
    # 然后返回由这些元组组成的列表。
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary



def divide_sentences_by_batch(data,batch_words):
    '''
    将语料按照batch_words的大小进行分
    data：以单词index形式保存的原始语料
    '''
    
    #train_data = list()  # 保存已经训练的语料，用于训练停止的判断
    trian_sentences_batch = list()
    cnt = 0
    while cnt < len(data):
    # 当train_data的长度与data相等说明所有句子都已经训练完成
        sentence_batch = list()
        batch_words_size = 0
        # 判断语料库中是否还有没有训练的语料
        while cnt < len(data):
            # 取出一个句子
            
            if(len(data[cnt]) > batch_words):
                sentence_tmp = data[cnt][0:batch_words-1]
            else:
                sentence_tmp = data[cnt]
            
            #print()
            if len(sentence_tmp) + batch_words_size <= batch_words:
                sentence_batch.append(sentence_tmp)
                batch_words_size += len(sentence_tmp)
                #train_data.append(sentence_tmp)
                cnt += 1
                if cnt % 100 == 0:
                    print("%d sentences have divide into batches." % cnt)
            else:
                break
        
        trian_sentences_batch.append(sentence_batch)
        #print("train_data:",train_data)
        #print("sentence_batch:",sentence_batch,'\n')
    return trian_sentences_batch








