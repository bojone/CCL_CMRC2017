#! -*- coding:utf-8 -*-

import codecs
import re
import os
import numpy as np

def split_data(text):
    words = re.split('[ \n]+', text)
    idx = words.index('XXXXX')
    return words[:idx],words[idx+1:]

print u'正在读取训练语料...'
train_x = codecs.open('../CMRC2017_train/train.doc_query', encoding='utf-8').read()
train_x = re.split('<qid_.*?\n', train_x)[:-1]
train_x = ['\n'.join([l.split('||| ')[1] for l in re.split('\n+', t) if l.split('||| ')[0]]) for t in train_x]
train_x = [split_data(l) for l in train_x]

train_y = codecs.open('../CMRC2017_train/train.answer', encoding='utf-8').read()
train_y = train_y.split('\n')[:-1]
train_y = [l.split('||| ')[1] for l in train_y]

print u'正在读取验证语料...'
valid_x = codecs.open('../CMRC2017_cloze_valid/cloze.valid.doc_query', encoding='utf-8').read()
valid_x = re.split('<qid_.*?\n', valid_x)[:-1]
valid_x = ['\n'.join([l.split('||| ')[1] for l in re.split('\n+', t) if l.split('||| ')[0]]) for t in valid_x]
valid_x = [split_data(l) for l in valid_x]

valid_y = codecs.open('../CMRC2017_cloze_valid/cloze.valid.answer', encoding='utf-8').read()
valid_y = valid_y.split('\n')[:-1]
valid_y = [l.split('||| ')[1] for l in valid_y]

word_size = 128
if os.path.exists('model.config'): #如果有则读取配置信息
    id2word,word2id,embedding_array = pickle.load(open('model.config'))
else: #如果没有则重新训练词向量
    import jieba
    import codecs
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    from gensim.models import Word2Vec
    print u'正在对添加语料进行分词...'
    additional = codecs.open('../additional.txt', encoding='utf-8').read().split('\n') #自行从网上爬的童话语料
    additional = map(lambda s: jieba.lcut(s, HMM=False), additional)
    class data_for_word2vec: #用迭代器将三个语料整合起来
        def __iter__(self):
            for x in train_x:
                yield x[0]
                yield x[1]
            for x in valid_x:
                yield x[0]
                yield x[1]
            for x in additional:
                yield x
    word2vec = Word2Vec(data_for_word2vec(), size=word_size, min_count=2, sg=2, negative=10, iter=10)
    word2vec.save('word2vec_tk')
    from collections import defaultdict
    id2word = {i+1:j for i,j in enumerate(word2vec.wv.index2word)}
    word2id = defaultdict(int, {j:i for i,j in id2word.items()})
    embedding_array = np.array([word2vec[id2word[i+1]] for i in range(len(id2word))])
    pickle.dump([id2word,word2id,embedding_array], open('model.config','w'))

import tensorflow as tf

padding_vec = tf.Variable(tf.random_uniform([1, word_size], -0.05, 0.05)) #只对填充向量进行训练，其余向量保持word2vec的结果
embeddings = tf.constant(embedding_array, dtype=tf.float32)
embeddings = tf.concat([padding_vec,embeddings], 0)

L_context = tf.placeholder(tf.int32, shape=[None,None])
L_context_length = tf.placeholder(tf.int32, shape=[None])
R_context = tf.placeholder(tf.int32, shape=[None,None])
R_context_length = tf.placeholder(tf.int32, shape=[None])

L_context_vec = tf.nn.embedding_lookup(embeddings, L_context)
R_context_vec = tf.nn.embedding_lookup(embeddings, R_context)

def add_brnn(inputs, rnn_size, seq_lens, name): #定义单层双向LSTM，上下文公用参数，分别过LSTM然后拼接
    rnn_cell_fw = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    rnn_cell_bw = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    outputs = []
    with tf.variable_scope(name_or_scope=name) as vs:
        for input,seq_len in zip(inputs,seq_lens):
            outputs.append(tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, input, sequence_length=seq_len, dtype=tf.float32))
            vs.reuse_variables()
    return [tf.concat(o[0],2) for o in outputs], [o[1] for o in outputs]

[L_outputs,R_outputs],[L_final_state,R_final_state] = add_brnn([L_context_vec,R_context_vec], word_size, [L_context_length,R_context_length], name='LSTM_1')
[L_outputs,R_outputs],[L_final_state,R_final_state] = add_brnn([L_outputs,R_outputs], word_size, [L_context_length,R_context_length], name='LSTM_2')

L_context_mask = (1-tf.cast(tf.sequence_mask(L_context_length), tf.float32))*(-1e12) #对填充位置进行mask，注意这里是softmax之前的mask，所以mask不是乘以0，而是减去1e12
R_context_mask = (1-tf.cast(tf.sequence_mask(R_context_length), tf.float32))*(-1e12)
context_mask = tf.concat([L_context_mask,R_context_mask], 1)

outputs = tf.concat([L_outputs,R_outputs], 1)
final_state = (tf.concat([L_final_state[0][1], L_final_state[1][1]], 1) + tf.concat([R_final_state[0][1], R_final_state[1][1]], 1))/2 #双向拼接、上下文取平均，得到encode向量
attention = context_mask + tf.matmul(outputs, tf.expand_dims(final_state, 2))[:,:,0] #encode向量与每个时间步状态向量做内积，然后mask，然后softmax
sample_labels = tf.placeholder(tf.float32, shape=[None,None])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=sample_labels, logits=attention))
pred = tf.nn.softmax(attention)

train_step = tf.train.AdamOptimizer().minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

train_x = [([word2id[i] for i in j[0]] if j[0] else [0], [word2id[i] for i in j[1]] if j[1] else [0]) for j in train_x] #词序列ID化
train_y = [word2id[i] for i in train_y]
valid_x = [([word2id[i] for i in j[0]] if j[0] else [0], [word2id[i] for i in j[1]] if j[1] else [0]) for j in valid_x]
valid_y = [word2id[i] for i in valid_y]

def construct_sample(x, y, i):
    return x[i][0], x[i][1], y[i]

train_x = [construct_sample(train_x, train_y, i) for i in range(len(train_x))] #输入输出配对，构成训练样本
valid_x = [construct_sample(valid_x, valid_y, i) for i in range(len(valid_x))]

batch_size = 160
def generate_batch_data(data, batch_size): #生成单个batch
    np.random.shuffle(data)
    batch = []
    for x in data:
        batch.append(x)
        if len(batch) == batch_size:
            l0 = [len(x[0]) for x in batch]
            l1 = [len(x[1]) for x in batch]
            x0 = np.array([x[0]+[0]*(max(l0)-len(x[0])) for x in batch])
            x1 = np.array([x[1]+[0]*(max(l1)-len(x[1])) for x in batch])
            x2 = np.array([[x[2]] for x in batch])
            y = (np.hstack([x0,x1])==x2).astype(np.float32)
            yield (x0,
                   x1,
                   y/y.sum(axis=1).reshape((-1,1)),
                   np.array(l0),
                   np.array(l1),
                   x2
                  )
            batch = []
    if batch:
        l0 = [len(x[0]) for x in batch]
        l1 = [len(x[1]) for x in batch]
        x0 = np.array([x[0]+[0]*(max(l0)-len(x[0])) for x in batch])
        x1 = np.array([x[1]+[0]*(max(l1)-len(x[1])) for x in batch])
        x2 = np.array([[x[2]] for x in batch])
        y = (np.hstack([x0,x1])==x2).astype(np.float32)
        yield (x0,
               x1,
               y/y.sum(axis=1).reshape((-1,1)),
               np.array(l0),
               np.array(l1),
               x2
              )
        batch = []

import datetime
import json

epochs = 30
saver = tf.train.Saver()
if not os.path.exists('./tk'):
    os.mkdir('./tk')
try:
    saver.restore(sess, './tk/tk_highest.ckpt')
except:
    pass

def cumsum_proba(x, y): #对相同项的概率进行合并
    tmp = {}
    for i,j in zip(x, y):
        if i in tmp:
            tmp[i] += j
        else:
            tmp[i] = j
    return tmp.keys()[np.argmax(tmp.values())]

highest_acc = 0.
train_log = {'loss':[], 'accuracy':[]}
for e in range(epochs):
    train_data = list(generate_batch_data(train_x, batch_size))
    count = 0
    batch = 0
    for x in train_data:
        if batch % 10 == 0:
            loss_ = sess.run(loss, feed_dict={L_context:x[0], R_context:x[1], sample_labels:x[2], L_context_length:x[3], R_context_length:x[4]})
            print '%s, epoch %s, trained on %s samples, loss: %s'%(datetime.datetime.now(), e+1, count, loss_)
            saver.save(sess, './tk/tk_%s.ckpt'%e) #每个epoch保存一次
            train_log['loss'].append(float(loss_))
            json.dump(train_log, open('train.log', 'w'))
        sess.run(train_step, feed_dict={L_context:x[0], R_context:x[1], sample_labels:x[2], L_context_length:x[3], R_context_length:x[4]})
        if batch % 100 == 0:
            valid_data = list(generate_batch_data(valid_x, batch_size))
            r = 0.
            for x in valid_data:
                p = sess.run(pred, feed_dict={L_context:x[0], R_context:x[1], sample_labels:x[2], L_context_length:x[3], R_context_length:x[4]})
                w = np.hstack([x[0],x[1]])
                r += (np.array([cumsum_proba(s,t) for s,t in zip(w, p)]) == x[5].reshape(-1)).sum()
            acc = r/len(valid_x)
            print '%s, valid accuracy %s'%(datetime.datetime.now(), acc)
            train_log['accuracy'].append(acc)
            if highest_acc <= acc:
                highest_acc = acc
                saver.save(sess, './tk/tk_highest.ckpt') #历史最好也保存一次
        batch += 1
        count += len(x[0])
