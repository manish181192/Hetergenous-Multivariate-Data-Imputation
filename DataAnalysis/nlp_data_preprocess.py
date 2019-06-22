import pandas as pd
import numpy as np
import config
import math
from nltk.tokenize import word_tokenize
datapath = "/home/manish/ML/Rutgers/Project/ML3/ML3/ML3AllSites.csv"

df = pd.read_csv(datapath)
# print(list(df))
col = "lowpower"
df = df[[col]]


vocab = {}
reverse_vocab = {}
vocab["<OOV>"] = 0 # out of vocab(new words)
vocab["<START>"] = 1 # start of sentence
vocab["<STOP>"] = 2 # end of sentence

reverse_vocab[0] = "<BLANK>"
reverse_vocab[1] = "<START>"
reverse_vocab[2] = "<STOP>"

word_id =3
max_seq_length = -1
for id, text in enumerate(df[col]):

    if pd.isnull(text):
        continue
    words = word_tokenize(text)
    if len(words) > max_seq_length:
        max_seq_length = len(words)
    for word in words:
        if word not in vocab.keys():
            vocab[word] = word_id
            reverse_vocab[word_id] = word
            word_id+=1

print("Max Sequence : {}".format(max_seq_length))
print("Vocab_size: {}".format(len(vocab)))

data_size = len(df[col])
text_numpy = np.zeros(shape=[data_size, max_seq_length+2], dtype= int)
for id, text in enumerate(df[col]):
    if pd.isnull(text):
        continue
    words = word_tokenize(text)
    text_ids = []
    text_numpy[id, 0] = vocab["<START>"]
    word_id = 1
    for word in words:
        text_numpy[id, word_id] = vocab[word]
        word_id+=1
    text_numpy[id, word_id] = vocab["<STOP>"]

if False:
    print("Saving numpy files ...")
    np.save("vocab_dir/"+col, vocab)
    np.save("nlp_reverse_vocab_dir/"+col, reverse_vocab)
    np.save("nlp_processed_data/"+col, text_numpy)
    print("Saved")