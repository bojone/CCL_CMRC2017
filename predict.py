#! -*- coding:utf-8 -*-

import pickle
import numpy as np

id2word,word2id,embedding_array = pickle.load(open('model.config'))
word_size = embedding_array.shape[1]

import tensorflow as tf

padding_vec = tf.Variable(tf.random_uniform([1, word_size], -0.05, 0.05))
embeddings = tf.constant(embedding_array, dtype=tf.float32)
embeddings = tf.concat([padding_vec,embeddings], 0)

L_context = tf.placeholder(tf.int32, shape=[None,None])
L_context_length = tf.placeholder(tf.int32, shape=[None])
R_context = tf.placeholder(tf.int32, shape=[None,None])
R_context_length = tf.placeholder(tf.int32, shape=[None])

L_context_vec = tf.nn.embedding_lookup(embeddings, L_context)
R_context_vec = tf.nn.embedding_lookup(embeddings, R_context)

def add_brnn(inputs, rnn_size, seq_lens, name):
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

L_context_mask = (1-tf.cast(tf.sequence_mask(L_context_length), tf.float32))*(-1e12)
R_context_mask = (1-tf.cast(tf.sequence_mask(R_context_length), tf.float32))*(-1e12)
context_mask = tf.concat([L_context_mask,R_context_mask], 1)

outputs = tf.concat([L_outputs,R_outputs], 1)
final_state = (tf.concat([L_final_state[0][1], L_final_state[1][1]], 1) + tf.concat([R_final_state[0][1], R_final_state[1][1]], 1))/2
attention = context_mask + tf.matmul(outputs, tf.expand_dims(final_state, 2))[:,:,0]
sample_labels = tf.placeholder(tf.float32, shape=[None,None])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=sample_labels, logits=attention))
pred = tf.nn.softmax(attention)

train_step = tf.train.AdamOptimizer().minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

saver = tf.train.Saver()
saver.restore(sess, './tk/tk_highest.ckpt')

import re
def split_data(text):
    words = re.split('[ \n]+', text)
    idx = words.index('XXXXX')
    return words[:idx],words[idx+1:]

def cumsum_proba(x, y):
    tmp = {}
    for i,j in zip(x, y):
        if i in tmp:
            tmp[i] += j
        else:
            tmp[i] = j
    return tmp.keys()[np.argmax(tmp.values())]

def predict(text): #输入的text为字符串，用空格隔开分词结果，待填空位置用XXXXX表示
    text = split_data(text)
    text = [word2id[i] for i in text[0]] if text[0] else [0], [word2id[i] for i in text[1]] if text[1] else [0]
    p = sess.run(pred, feed_dict={L_context:[text[0]], R_context:[text[1]], L_context_length:[len(text[0])], R_context_length:[len(text[1])]})
    return id2word.get(cumsum_proba(text[0]+text[1], p[0]),' ')


if __name__ == '__main__':

    import codecs
    import os
    import sys

    vaild_name = sys.argv[1]
    output_name = sys.argv[2]

    text = codecs.open(vaild_name, encoding='utf-8').read()
    valid_x = re.split('<qid_.*?\n', text)[:-1]
    valid_x = ['\n'.join([l.split('||| ')[1] for l in re.split('\n+', t) if l.split('||| ')[0]]) for t in valid_x]
    valid_x = [split_data(l) for l in valid_x]
    valid_x = [([word2id[i] for i in j[0]] if j[0] else [0], [word2id[i] for i in j[1]] if j[1] else [0]) for j in valid_x]

    batch_size = 160
    def generate_batch_data(data, batch_size):
        batch = []
        for x in data:
            batch.append(x)
            if len(batch) == batch_size:
                l0 = [len(x[0]) for x in batch]
                l1 = [len(x[1]) for x in batch]
                x0 = np.array([x[0]+[0]*(max(l0)-len(x[0])) for x in batch])
                x1 = np.array([x[1]+[0]*(max(l1)-len(x[1])) for x in batch])
                yield (x0,
                       x1,
                       np.array(l0),
                       np.array(l1),
                      )
                batch = []
        if batch:
            l0 = [len(x[0]) for x in batch]
            l1 = [len(x[1]) for x in batch]
            x0 = np.array([x[0]+[0]*(max(l0)-len(x[0])) for x in batch])
            x1 = np.array([x[1]+[0]*(max(l1)-len(x[1])) for x in batch])
            yield (x0,
                   x1,
                   np.array(l0),
                   np.array(l1),
                  )
            batch = []

    valid_data = list(generate_batch_data(valid_x, batch_size))
    valid_result = []
    for x in valid_data:
        p = sess.run(pred, feed_dict={L_context:x[0], R_context:x[1], L_context_length:x[2], R_context_length:x[3]})
        w = np.hstack([x[0],x[1]])
        valid_result.extend(np.array([cumsum_proba(s,t) for s,t in zip(w, p)]))

    #生成讯飞杯要求的评测格式
    names = re.findall('<qid_\d+>', text)
    s = '\n'.join(names[i]+' ||| '+id2word.get(j,' ') for i,j in enumerate(valid_result))
    with codecs.open(output_name, 'w', encoding='utf-8') as f:
        f.write(s)
