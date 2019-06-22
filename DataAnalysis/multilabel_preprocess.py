import pandas as pd
import numpy as np
import config
import math
from nltk.tokenize import word_tokenize
datapath = "/home/manish/ML/Rutgers/Project/ML3/ML3/ML3AllSites.csv"

df = pd.read_csv(datapath)
# print(list(df))
col = "anagrams2"
df = df[[col]]


vocab = {}
reverse_vocab = {}
word_id=0
data_list = []
for id, text in enumerate(df[col]):
    if id==965:
        print()
    if pd.isnull(text):
        data_list.append([])
        continue
    text = text.lower()
    words = word_tokenize(text)
    while ',' in words: words.remove(',')
    data_list.append(words)
    for word in words:
        if word not in vocab.keys():
            vocab[word] = word_id
            reverse_vocab[word_id] = word
            word_id+=1

print("Vocab_size: {}".format(len(vocab)))
print(vocab)
data_size = len(df[col])
text_numpy = np.zeros(shape=[data_size,len(vocab)], dtype= int)
for id, text_list in enumerate(data_list):

    if len(text_list)==0:
        continue
    words = text_list
    for word in words:
        text_numpy[id,vocab[word]] = 1

if False:
    print("Saving numpy files ...")
    np.save("vocab_dir/"+col, vocab)
    np.save("reverse_vocab_dir/"+col, reverse_vocab)
    np.save("multilabel_processed_data/"+col, text_numpy)
    print("Saved")