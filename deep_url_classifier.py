#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Copyright 2019, Seokjun Bu, Softcomputing LAB all rights reserved.
import sys
import os
import io
import json
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Lambda, concatenate, TimeDistributed, LSTM, BatchNormalization, Embedding, GlobalMaxPool2D, Dense, Conv2D, MaxPool2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, UpSampling2D, RepeatVector, TimeDistributed, Conv1D, MaxPool1D, UpSampling1D
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[2]:


# 하이퍼파라미터
random_state_seed = 11
n_benign, n_phishing = 45000, 15000
n_max_char, n_type_char = 100, 150
n_max_word, n_top_word = 32, 5000


# In[9]:





# In[15]:


# 스크립트 사용 예시 python ./deep_url_classifier.py http://www.goooglee.com
if __name__ == "__main__":
    url_input = sys.argv[1]
    url_input_len = len(url_input)
    # 문자수준 전처리
    unique_char_list = np.load('./resource/unique_char_list.npy')
    url_input_char = np.zeros((1, n_max_char))
    for c in range(n_max_char):
        if(c<url_input_len):
            url_input_char[0, c] = ord(url_input[c])
    url_input_char_cat = np.zeros((1, n_max_char, n_type_char))   
    for c in range(n_max_char):
        url_input_char_cat[0, c, [np.argwhere(unique_char_list==url_input_char[0, c])[0][0]]] = 1
    # 단어수준 전처리
    with open('./resource/tokenizer.json') as f:
        tkz = tokenizer_from_json(json.load(f))
    url_input_word = pad_sequences(tkz.texts_to_sequences([url_input]), n_max_word)
    # 모델 로드
    deep_url_classifier = load_model('./model/fusion.h5')
    # 출력
    output = deep_url_classifier.predict([url_input_char_cat, url_input_word])[0]
    prob_benign, prob_phishing = output[0], output[1]
    print("Benign: %.4f" %(prob_benign))
    print("Phishing: %.4f" %(prob_phishing))
    np.savetxt('./public/output.txt', output, fmt='%.4f', delimiter=',')


# In[9]:


# # 데이터셋 로딩
# dataset_benign, dataset_phishing = pd.read_csv('./url_0.csv', header=None, skiprows=0).sample(n=n_benign, random_state=random_state_seed), pd.read_csv('./url_1.csv', header=None, skiprows=0).sample(n=n_phishing, random_state=random_state_seed)
# dataset_benign, dataset_phishing = dataset_benign[1].values, dataset_phishing[1].values
# # unique_char_list.npy 만들기
# dataset_char_seq = np.zeros((n_benign + n_phishing, n_max_char))
# for r in range(dataset_benign.shape[0]):
#     length_buffer = len(dataset_benign[r])
#     for c in range(n_max_char):
#         if(c<length_buffer):
#             dataset_char_seq[r, c] = ord(dataset_benign[r][c])
#         else:
#             break
# for r in range(dataset_phishing.shape[0]):
#     length_buffer = len(dataset_phishing[r])
#     for c in range(n_max_char):
#         if(c<length_buffer):
#             dataset_char_seq[r+n_benign, c] = ord(dataset_phishing[r][c])
#         else:
#             break
# unique_char_list = np.unique(dataset_char_seq).astype(int)
# np.save('./resource/unique_char_list', unique_char_list)
# # Tokenizer 만들기
# dataset_word_seq = []
# for r in range(dataset_benign.shape[0]):
#     dataset_word_seq.append(dataset_benign[r])
# for r in range(dataset_phishing.shape[0]):
#     dataset_word_seq.append(dataset_phishing[r])
# dataset_word_seq = np.array(dataset_word_seq)
# print(dataset_word_seq)
# tkz = Tokenizer(n_top_word, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split='/')
# tkz.fit_on_texts(dataset_word_seq)
# tkz_json = tkz.to_json()
# with io.open('./resource/tokenizer.json', 'w', encoding='utf-8') as f:
#     f.write(json.dumps(tkz_json, ensure_ascii=False))


# In[ ]:




