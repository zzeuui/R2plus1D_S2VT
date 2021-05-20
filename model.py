#-*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import numpy as np
import os

from tensorflow._api.v1 import summary
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt
import sys
import time
import cv2
#from keras.preprocessing import sequence
import pdb
import math
import struct
import datetime

from utils.score import COCOScorer

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
script_dir = os.path.dirname(os.path.abspath(__file__))
class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_step, n_video_lstm_step, n_caption_lstm_step, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_step = n_lstm_step
        self.n_video_lstm_step = n_video_lstm_step
        self.n_caption_lstm_step = n_caption_lstm_step

        self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

        self.lstm1 = tf.contrib.rnn.BasicLSTMCell(dim_hidden)
        self.lstm2 = tf.contrib.rnn.BasicLSTMCell(dim_hidden)

        self.encode_image_W = tf.Variable(tf.random_uniform([dim_image, dim_hidden], -0.1,0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def build_model(self):
        print("")
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step])

        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_lstm_step+1])
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step+1])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_step, self.dim_hidden])

        state1 = self.lstm1.zero_state(self.batch_size, dtype=tf.float32)
        state2 = self.lstm2.zero_state(self.batch_size, dtype=tf.float32)
        padding = tf.zeros([self.batch_size, self.dim_hidden])

        probs = []
        loss = 0.0

        ###### encoding
        with tf.variable_scope("encoding") as scope:
            for i in range(0, self.n_video_lstm_step): ### Phase 1: only read frames
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                with tf.variable_scope("LSTM1") as scope:
                    (output1, state1) = self.lstm1(image_emb[:,i,:], state1)

                with tf.variable_scope("LSTM2") as scope:
                    (output2, state2) = self.lstm2(tf.concat([padding, output1], 1), state2)

        ##### decoding
        with tf.variable_scope("encoding") as scope:
            for i in range(0, self.n_caption_lstm_step): ### Phase 2: only generate captions
                current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:,i])

                tf.get_variable_scope().reuse_variables()

                with tf.variable_scope("LSTM1") as scope:
                    (output1, state1) = self.lstm1(padding, state1)

                with tf.variable_scope("LSTM2") as scope:
                    (output2, state2) = self.lstm2(tf.concat([current_embed, output1], 1), state2)

                labels = tf.expand_dims(caption[:, i+1], 1)
                indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
                concated = tf.concat([indices, labels], 1)
                onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

                logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logit_words)
                cross_entropy = cross_entropy * caption_mask[:,i]
                probs.append(logit_words)

                current_loss = tf.reduce_sum(cross_entropy)/self.batch_size
                loss = loss + current_loss

        return loss, video, video_mask, caption, caption_mask, probs


    def build_generator(self):
        video = tf.placeholder(tf.float32, [1,self.n_video_lstm_step, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [1, self.n_video_lstm_step])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_video_lstm_step, self.dim_hidden])

        state1 = self.lstm1.zero_state(1, dtype=tf.float32)
        state2 = self.lstm2.zero_state(1, dtype=tf.float32)
        padding = tf.zeros([1, self.dim_hidden])

        generated_words = []
        probs = []
        embeds = []

        with tf.variable_scope("encoding") as scope:
            for i in range(0, self.n_video_lstm_step):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                with tf.variable_scope("LSTM1"):
                    (output1, state1) = self.lstm1(image_emb[:, i, :], state1)

                with tf.variable_scope("LSTM2"):
                    (output2, state2) = self.lstm2(tf.concat([padding, output1], 1), state2)

        with tf.variable_scope("encoding") as scope:
            for i in range(0, self.n_caption_lstm_step):
                tf.get_variable_scope().reuse_variables()
                if i == 0:
                    current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))

                with tf.variable_scope("LSTM1"):
                    (output1, state1) = self.lstm1(padding, state1)

                with tf.variable_scope("LSTM2"):
                    (output2, state2) = self.lstm2(tf.concat([current_embed, output1], 1), state2)

                logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
                max_prob_index = tf.argmax(logit_words, 1)[0]
                generated_words.append(max_prob_index)
                probs.append(logit_words)

                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                current_embed = tf.expand_dims(current_embed, 0)

                embeds.append(current_embed)

        return video, video_mask, generated_words, probs, embeds


#### global parameters
video_path = os.path.join(script_dir, 'data/youtube_videos')
video_feat_path = os.path.join(script_dir, 'data/features/r2plus1d_34_476')
video_data_path = os.path.join(script_dir, 'data/msvd/video_corpus_476.csv')
model_path = os.path.join(script_dir, 'data/checkpoints/r_plus_476')

#true 1, false 0
load_model = 0

print('video_feat_path: ', video_feat_path)
print('model_path: ', model_path)

video_feat_path_num = video_feat_path.split('/')
video_feat_path_num = len(video_feat_path_num)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'

print("Log dit >", train_log_dir)

train_summary_writer = tf.summary.FileWriter(train_log_dir)
test_summary_writer = tf.summary.FileWriter(test_log_dir)

# try:
#     if not(os.path.isdir(model_path)):
#         os.makedirs(os.path.join(model_path))
# except OSError as e:
#     if e.errno != e.errno.EEXIST:
#         print("failed to create model save edirectory")
#         raise


#### train parameters
dim_image = 359
dim_hidden = 800

n_video_lstm_step = 20
n_caption_lstm_step = 20
n_frame_step = 20

n_epochs = 100
batch_size = 32
learning_rate = 0.00001

def get_video_data(video_data_path, video_feat_path, train_ratio=0.9):
    video_data = pd.read_csv(video_data_path, sep=',')
    video_data = video_data.dropna()
    video_data['Start'] = video_data['Start'].astype(int)
    video_data['End'] = video_data['End'].astype(int)
    video_data = video_data[video_data['Language'] == 'English']
    video_data['video_path'] = video_data.apply(lambda row: row['VideoID']+'_'+str(row['Start'])+'_'+str(row['End'])+'.avi.npy', axis=1)
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
    #video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
    video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]

    unique_filenames = video_data['video_path'].unique()

    train_len = int(len(unique_filenames)*train_ratio)

    train_vids = []
    test_vids =[]

    train_file_list = os.listdir(os.path.join(script_dir,'data/features/train'))
    test_file_list = os.listdir(os.path.join(script_dir,'data/features/test'))
    for i in train_file_list:
        train_vids.append(os.path.join(script_dir,'data/features/r2plus1d_34_476/'+i))
    for i in test_file_list:
        test_vids.append(os.path.join(script_dir,'data/features/r2plus1d_34_476/'+i))
    #train_vids = unique_filenames[:train_len]
    #test_vids = unique_filenames[train_len:]

    train_data = video_data[video_data['video_path'].map(lambda x: x in train_vids)]
    test_data = video_data[video_data['video_path'].map(lambda x: x in test_vids)]

    return train_data, test_data

def preProBuildWordVocab(sentence_iterator, word_count_threshold=5):
    print('preprocessing word counts and creating vocab based on word count threshold %d' % word_count_threshold)
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('filtered words from %d to %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '<pad>'
    ixtoword[1] = '<bos>'
    ixtoword[2] = '<eos>'
    ixtoword[3] = '<unk>'

    wordtoix = {}
    wordtoix['<pad>'] = 0
    wordtoix['<bos>'] = 1
    wordtoix['<eos>'] = 2
    wordtoix['<unk>'] = 3

    for idx, w in enumerate(vocab):
        wordtoix[w] = idx + 4
        ixtoword[idx+4] = w

    word_counts['<pad>'] = nsents
    word_counts['<bos>'] = nsents
    word_counts['<eos>'] = nsents
    word_counts['<unk>'] = nsents

    bias_init_vector = np.array([1.0 * word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

    return wordtoix, ixtoword, bias_init_vector

def train():
    global video_feat_path
    train_data, test_data = get_video_data(video_data_path, video_feat_path, 0.9)
    test_videos = test_data['video_path'].unique()
    train_captions = train_data['Description'].values
    loss_val = 0
    captions_list = list(train_captions)
    captions = np.array(captions_list, dtype=np.object)

    captions = list(map(lambda x: x.replace('.', ''), captions))
    captions = list(map(lambda x: x.replace(',', ''), captions))
    captions = list(map(lambda x: x.replace('"', ''), captions))
    captions = list(map(lambda x: x.replace('\n', ''), captions))
    captions = list(map(lambda x: x.replace('?', ''), captions))
    captions = list(map(lambda x: x.replace('!', ''), captions))
    captions = list(map(lambda x: x.replace('\\', ''), captions))
    captions = list(map(lambda x: x.replace('/', ''), captions))

    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=0)


    np.save(os.path.join(script_dir,'data/wordtoix'), wordtoix)
    np.save(os.path.join(script_dir,'data/ixtoword'), ixtoword)
    np.save(os.path.join(script_dir,'data/bias_init_vector'), bias_init_vector)

    ixtoword = pd.Series(ixtoword)

    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(wordtoix),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_step=n_frame_step,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step,
            bias_init_vector=bias_init_vector)

    tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_probs = model.build_model()
    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()

    config = tf.ConfigProto()
    sess = tf.InteractiveSession(config=config)

    # tensorflow version 1.3
    saver = tf.train.Saver(max_to_keep=200)
    #train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_loss)
    #train_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(tf_loss)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)

    tf.global_variables_initializer().run()

    #load model
    if load_model == 1:
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt != None:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('model load from ...', model_path)

    start_time = time.time()
    for epoch in range(1, n_epochs+1):
        index = list(train_data.index)
        np.random.shuffle(index)
        train_data = train_data.loc[index]

        current_train_data = train_data.groupby('video_path').apply(lambda x: x.iloc[np.random.choice(len(x))])
        current_train_data = current_train_data.reset_index(drop=True)

        print('Epoch:', epoch, 'start learning')
        for start, end in zip(
                range(0, len(current_train_data), batch_size),
                range(batch_size, len(current_train_data), batch_size)):

            current_batch = current_train_data[start:end]
            current_videos = current_batch['video_path'].values

            current_feats = np.zeros((batch_size, n_video_lstm_step, dim_image))
            current_feats_vals = list(map(lambda vid: np.load(vid), current_videos))

            current_video_masks = np.zeros((batch_size, n_video_lstm_step))

            for ind, feat in enumerate(current_feats_vals):
                current_feats[ind][:len(current_feats_vals[ind])] = feat
                current_video_masks[ind][:len(current_feats_vals[ind])] = 1

            current_captions = current_batch['Description'].values
            current_captions = list(map(lambda x: '<bos> ' + x, current_captions))
            current_captions = list(map(lambda x: x.replace('.', ''), current_captions))
            current_captions = list(map(lambda x: x.replace(',', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('"', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('\n', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('?', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('!', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('\\', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('/', ''), current_captions))

            for idx, each_cap in enumerate(current_captions):
                word = each_cap.lower().split(' ')
                if len(word) < n_caption_lstm_step:
                    current_captions[idx] = current_captions[idx] + ' <eos>'
                else:
                    new_word = ''
                    for i in range(n_caption_lstm_step-1):
                        new_word = new_word + word[i] + ' '
                    current_captions[idx] = new_word + '<eos>'

            current_caption_ind = []
            for cap in current_captions:
                current_word_ind = []
                for word in cap.lower().split(' '):
                    if word in wordtoix:
                        current_word_ind.append(wordtoix[word])
                    else:
                        current_word_ind.append(wordtoix['<unk>'])
                current_caption_ind.append(current_word_ind)

            current_caption_matrix = tf.keras.preprocessing.sequence.pad_sequences(current_caption_ind, padding='post', maxlen=n_caption_lstm_step)
            current_caption_matrix = np.hstack([current_caption_matrix, np.zeros([len(current_caption_matrix), 1])]).astype(int)
            current_caption_masks = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array(list(map(lambda x: (x!=0).sum() + 1, current_caption_matrix)))

            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1

            if load_model == 0:
                probs_val = sess.run(tf_probs, feed_dict={tf_video:current_feats, tf_caption:current_caption_matrix})

                _, loss_val = sess.run([train_op, tf_loss],
                        feed_dict={tf_video: current_feats, tf_video_mask: current_video_masks, tf_caption: current_caption_matrix,
                            tf_caption_mask: current_caption_masks
                            })

                print('learning batch:', start/batch_size, '/', len(current_train_data)/batch_size, '\r', end='')
        print('loss:', loss_val, 'Elapsed time:', str((time.time()-start_time)))

        if 0 == epoch%10:
            gts = {}
            ref = {}
            IDs = []
            
            for idx, video_feat_path in enumerate(test_videos):
                video_feat = np.load(video_feat_path)[None, ...]

                if video_feat.shape[1] == n_frame_step:
                    video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
                else:
                    continue

                generated_word_index = sess.run(caption_tf, feed_dict={video_tf: video_feat, video_mask_tf: video_mask})
                generated_words = ixtoword[generated_word_index]

                punctuation = np.argmax(np.array(generated_words) == '<eos>')+1
                generated_words = generated_words[:punctuation]

                generated_sentence = ' '.join(generated_words)
                generated_sentence = generated_sentence.replace('<bos> ', '')
                generated_sentence = generated_sentence.replace(' <eos>', '')
                video_name = video_feat_path.split('/')[video_feat_path_num].split('.')[0]

                video_r = ""

                for i in range(len(video_name.split('_'))-2):
                    if i == 0:
                        video_r += video_name.split('_')[i]
                    else:
                        video_r += "_" + video_name.split('_')[i]

                video_start = video_name.split('_')[-2]
                video_end = video_name.split('_')[-1]

                sub_data = test_data[test_data['Language'] == 'English']
                sub_data = sub_data[sub_data['End'] == int(video_end)]
                sub_data = sub_data[sub_data['Start'] == int(video_start)]
                sub_data = sub_data[sub_data["VideoID"] == video_r]

                ref_list = []

                for i in sub_data['Description']:
                    stay_ref = {u'image_id': idx, u'caption': i}
                    ref_list.append(stay_ref)

                ref[idx] = ref_list

                stay_gts = {u'image_id': idx, u'caption': generated_sentence}
                stay_gts = [stay_gts]
                gts[idx] = stay_gts
                

                IDs.append(idx)
                #print("video name:", video_name)
                #print("generated sentence:", generated_sentence)
                #print("ref list:", ref_list)

            scorer = COCOScorer()
            score = scorer.score(ref, gts, IDs)

            bleu1 = score['Bleu_1']
            bleu1 = math.ceil(bleu1*10000)/100

            bleu2 = score['Bleu_2']
            bleu2 = math.ceil(bleu2*10000)/100

            bleu3 = score['Bleu_3']
            bleu3 = math.ceil(bleu3 * 10000) / 100

            bleu4 = score['Bleu_4']
            bleu4 = math.ceil(bleu4 * 10000) / 100

            meteor = score['METEOR']
            meteor = int.from_bytes(meteor, byteorder='big')
            meteor = math.ceil(meteor * 10000) / 100

            rouge_l = score['ROUGE_L']
            rouge_l = int.from_bytes(rouge_l, byteorder='big')
            rouge_l = math.ceil(rouge_l * 10000) / 100

            cider = score['CIDEr']
            cider = int.from_bytes(cider, byteorder='big')
            cider = math.ceil(cider * 10000) / 100

            bleu1Scalar = tf.summary.scalar('bleu1Scalar', meteor)
            bleu2Scalar = tf.summary.scalar('bleu2Scalar', meteor)
            bleu3Scalar = tf.summary.scalar('bleu3Scalar', meteor)

            merged = tf.summary.merge_all()

            summary, _ = sess.run([merged, tf.convert_to_tensor(loss_val, dtype=tf.float32)])
            train_summary_writer.add_summary(summary, 0)
            train_summary_writer.close()

            print("Epoch ", epoch, "is done. Saving the model in ", model_path)
            saver.save(sess, os.path.join(model_path,'loss_' + str(loss_val) +
                                          '_B1_' + str(bleu1) + '_B2_' + str(bleu2) + '_B3_' + str(bleu3) + '_B4_' + str(bleu4)
                                          + 'M_' + str(meteor) + '_R_' + str(rouge_l) + '_C_' + str(cider)), global_step=epoch)

if __name__ == '__main__':
    train()
