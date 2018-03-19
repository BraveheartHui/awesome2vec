# readme

使用tensorflow实现Word2vec skip-gram模型

在tensorflow官方给出的word2vec_basic.py基础上稍作修改，原代码处理的是不分段落句子的语料（即整个语料是一整句）

修改后的代码能够处理多个句子组成的语料，对每一个句子进行处理，更新词向量和损失函数。
