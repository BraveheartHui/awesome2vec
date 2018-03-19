# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 08:31:03 2018

@author: hxh
"""

import w2v_preprocess as pre
import train_sg
import numpy as np
import tensorflow as tf
from six.moves import xrange
import math
import os
import sys
import argparse
from tempfile import gettempdir
from copy import deepcopy

from tensorflow.contrib.tensorboard.plugins import projector

# Give a folder path as an argument with '--log_dir' to save
# TensorBoard summaries. Default is a log folder in current directory.
# 保存log日志作为tensorboard可视化的数据来源
current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_dir',
    type=str,
    default=os.path.join(current_path, 'log'),
    help='The log directory for TensorBoard summaries.')
FLAGS, unparsed = parser.parse_known_args()





# =============================================================================
# step1 : 处理已分词的原始语料，保存单词表等相关数据
filename = 'corpus/train.txt'
#filename = 'shorttrain.dat'

sentences, raw_words = pre.read_data_to_sentences(filename)
# sentences 保存的是str类型的语料列表
# raw_words 为后面提供计数所用
print("sentences number: ",len(sentences))

# 选取频数大于min_count的单词，其余单词认定为UNK--Unknown,编号为0
#vocabulary_size = 50000
min_count = 5
data, count, dictionary, reverse_dictionary = pre.build_dataset(raw_words,sentences,min_count)
# data list 以单词序号的形式保存语料
# count list 保存每个词及对应的词频（词频>min_count）
# dictionary dict 保存每个对应的序号index，词频不足的统一index为0
# reverse_dictionary dict 将dictionary中的词与index位置对换
vocabulary_size = len(count)
sentence_avg_len = len(raw_words)/len(data)
print("average length of sentences: ",sentence_avg_len)
#print("data:",data)

#为了节约内存删除原始单词列表，打印出最高频出现的词汇及其数量
del sentences  # Hint to reduce memory.
del raw_words

# 输出部分测试
#print ('Most common words (+UNK)', count)



'''
# train_sg函数结果的输出测试
# 循环遍历每个batch中的句子
batch, labels = train_sg.train_batch_sg(data,skip_window=1)
for i in range(len(batch)):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
'''
'''
58 康小姐 -> 22 寮步镇
22 寮步镇 -> 58 康小姐
22 寮步镇 -> 23 莞樟路
23 莞樟路 -> 22 寮步镇
23 莞樟路 -> 59 石井
59 石井 -> 23 莞樟路
59 石井 -> 24 附近
24 附近 -> 59 石井
'''


#print(dictionary['张无忌'],dictionary['谢逊'],dictionary['赵敏'],dictionary['周芷若'])

# Step 2: Build and train a skip-gram model.
# 定义训练时的参数
#batch_words = 5000  # 定义一次迭代单词数量
embedding_size = 200
skip_window = 1
#num_skips = 2
num_sampled = sentence_avg_len/2



# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
# 选择随机验证集来考察其相近的单词是否与实际相符
valid_size = 5
valid_window = 30

# 返回一个长度为valid_size数值范围在[0,valid_window)之间
valid_examples = np.random.choice(valid_window, valid_size, replace = False)
for i in range(valid_size):
    print("valid_examples:", reverse_dictionary[valid_examples[i]])
#valid_examples = np.array([8,42,43,53])


#定义Skip-Gram Word2Vec模型的网络结构
print("Begin to construct Graph.")
graph = tf.Graph()

with graph.as_default():

  # Input data.
  with tf.name_scope('inputs'):
    # 保存一个batch内单词对的所有目标词
    train_inputs = tf.placeholder(tf.int32, shape=[None])
    # 保存一个batch内所有单词对中的上下文单词
    train_labels = tf.placeholder(tf.int32, shape=[None, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    with tf.name_scope('embeddings'):
      # random_uniform从均匀分布中输出随机值，生成值遵循[vocabulary_size, embedding_size]分布
      embeddings = tf.Variable(
          tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
      # 根据train_inputs中的内容，其在embeddings中的对应元素
      embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    with tf.name_scope('weights'):
      # 定义权重矩阵
      nce_weights = tf.Variable(
          tf.truncated_normal(
              [vocabulary_size, embedding_size],
              stddev=1.0 / math.sqrt(embedding_size)))
    with tf.name_scope('biases'):
      #定义偏置矩阵
      nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  # Explanation of the meaning of NCE loss:
  #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
  with tf.name_scope('loss'):
    loss = tf.reduce_mean(
        tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=train_labels,
            inputs=embed,
            num_sampled=num_sampled,
            num_classes=vocabulary_size))

  # Add the loss value as a scalar to summary.
  tf.summary.scalar('loss', loss)

  # Construct the SGD optimizer using a learning rate of 1.0.
  with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  # 计算cosine相似度
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  
  # 获取验证集中的词归一化向量，valid_dataset保存的是验证集单词的序号
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                            valid_dataset)
  # 计算验证集中每一个向量和所有其他向量的相似度
  # 返回的向量行数=验证集行数，列数=embeddings行数
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Merge all summaries.
  merged = tf.summary.merge_all()

  # 初始化所有的变量
  # Add variable initializer.
  init = tf.global_variables_initializer()

  # Create a saver.
  saver = tf.train.Saver()



# Step 3: Begin training.训练
step = 0
# 语料迭代的次数
iter_num = 5
data_temp = deepcopy(data)
# 根据迭代次数，对语料进行重复
for i in range(iter_num-1):
    data.extend(data_temp)


#print("Begin to divide sentences into batches.")
# sentences_batch = pre.divide_sentences_by_batch(data,batch_words)






with tf.Session(graph=graph) as session:
  #global num_sampled
  # Open a writer to write summaries.
  writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)
  
  # We must initialize all variables before we use them.
  # 首先初始化所有参数
  init.run()
  print('Initialized')
  
  # 损失函数定义为0
  average_loss = 0
  #for step in xrange(num_steps):
  for sentence in data:
    # 获得input目标单词与上下文单词的一一对应
    batch_inputs, batch_labels = train_sg.train_sg_pair(sentence, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
    # num_sampled = len(batch_inputs)/2
    # Define metadata variable.
    run_metadata = tf.RunMetadata()

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
    # Feed metadata variable to session for visualizing the graph in TensorBoard.
    _, summary, loss_val = session.run(
        [optimizer, merged, loss],
        feed_dict=feed_dict,
        run_metadata=run_metadata)
    average_loss += loss_val
    
    # Add returned summaries to writer in each step.
    writer.add_summary(summary, step)
    
    # Add metadata to visualize the graph for the last run.
    if step == (len(data) - 1):
      writer.add_run_metadata(run_metadata, 'step%d' % step)
    
    if step % 10 == 0:
      if step > 0:
        average_loss /= 10
        #print('num_sampled: ',num_sampled)
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
   
    
    if step % 50 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
    
    
    final_embeddings = normalized_embeddings.eval()
    step += 1
    
  # Write corresponding labels for the embeddings.
  with open(FLAGS.log_dir + '/metadata.tsv', 'w') as f:
    for i in xrange(vocabulary_size):
      f.write(reverse_dictionary[i] + '\n')

  # Save the model for checkpoints.
  saver.save(session, os.path.join(FLAGS.log_dir, 'model.ckpt'))

  # Create a configuration for visualizing embeddings with the labels in TensorBoard.
  config = projector.ProjectorConfig()
  embedding_conf = config.embeddings.add()
  embedding_conf.tensor_name = embeddings.name
  embedding_conf.metadata_path = os.path.join(FLAGS.log_dir, 'metadata.tsv')
  projector.visualize_embeddings(writer, config)

writer.close()













# Step 6: Visualize the embeddings.


# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.
# 词嵌入的可视化

'''
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(
        label,
        xy=(x, y),
        xytext=(5, 2),
        textcoords='offset points',
        ha='right',
        va='bottom')

  plt.savefig(filename)


try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(
      perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))

except ImportError as ex:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  print(ex)

 
'''









