# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 08:50:10 2018

train sg 句子序列中的窗口内的词和其上下文词成对提取出来


@author: hxh
"""

import numpy as np
import tensorflow as tf


def train_sg_pair(sentence, skip_window):
    '''
    train every pair of word in a single sentence. 
    Called internally from 'train_batch_sg()'.
    sentence:以单词id形式保存的语料离散（单独一句）
    '''
    target_list = list()
    context_list = list()
    # 先遍历句子中的每一个词
    for target_pos, target_word in enumerate(sentence):
        start = max(0, target_pos - skip_window)
        #end = max(len(sentence) + 1, target_pos + skip_window + 1)
        for pos, word in enumerate(sentence[start : target_pos + skip_window + 1]):
            if word != target_word:
                target_list.append(target_word)
                context_list.append(word)
    
    # batch保存的是target word，每个词出现的词数=窗口大小*2
    batch = np.ndarray(shape = len(target_list), buffer = np.array(target_list), dtype = np.int32)
    # labels保存的是与batch每个位置对应的context单词
    labels = np.ndarray(shape = (len(context_list), 1), buffer = np.array(context_list), dtype = np.int32)
    return batch,labels




def train_batch_sg(sentences, skip_window):
    '''
    Update skip-gram model by training on a sequence of sentences. 
    sentences：以单词index格式保存的句子序列
    skip_window：滑动窗口大小
    '''
    target_list = list()
    context_list = list()
    # print(sentences)
    #单独处理句子序列的每一个句子
    for sentence in sentences: 
        s_target_list,s_context_list = train_sg_pair(sentence, skip_window)
        target_list.extend(s_target_list)
        context_list.extend(s_context_list)
    # batch保存的是target word，每个词出现的词数=窗口大小*2
    batch = np.ndarray(shape = len(target_list), buffer = np.array(target_list), dtype = np.int32)
    # labels保存的是与batch每个位置对应的context单词
    labels = np.ndarray(shape = (len(context_list), 1), buffer = np.array(context_list), dtype = np.int32)
    return batch, labels
        
        
        
        
        
        
        
        
        
        
        