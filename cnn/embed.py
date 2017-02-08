import os
import cPickle as pickle
import numpy as np
import pandas as pd
from collections import defaultdict


# Set Constants
#############################################

DATA_PATH = "/Users/haohanchen/Dropbox/WORK/Research (now)/NN-vote/Data"
IN_PRETRAINED_VEC = "glove_840B_300d.txt"
IN_BILL = "cleaned_bill_house112.pickle"
IN_VOTE_BILL = "cleaned_bill_house112.pickle"
OUT_FILE = "bill112_embeded.pickle"
CV = 10 # Number of Folds
MIN_DF = 1 # Threshold document frequency above which word vector is assigned
K = 300 # Dimensions of word embedding

# Load raw text data
#############################################
with open(os.path.join(DATA_PATH, IN_BILL), 'r') as f:
    bills_raw = pickle.load(f)

for bill in bills_raw['main'][:20]:
	print bill[:100]


# Build dataset and vocabulary
#############################################

bills_tok = dict(main_tokenized = [], num_words = [])
vocab = defaultdict(float)
num_bills = len(bills_raw['main'])

i = 1
for bill in bills_raw['main']:
	bill_tok = bill.split(" ")
	words = set(bill_tok)
	for word in words:
		vocab[word] += 1
	bills_tok['main_tokenized'].append(bill_tok)
	bills_tok['num_words'].append(len(bill_tok))
	if i % 100 == 0:
		print "%d bills processed." % (i, )
	i += 1




# Match Pre-trained word2vec with dictionary
#############################################
print "Reading pre-trained wor2vec"

word_vecs = defaultdict(float)
with open(os.path.join(DATA_PATH, IN_PRETRAINED_VEC)) as f:
	i = 1
	for line in f.xreadlines():
		l = line.split()
		word = l[0]
		vec = l[1:(K+1)]
		if vocab.has_key(word):
			word_vecs[word] = vec
		if i % 2000 == 0:
			print "%d lines scanned" % (i, )
		i += 1

print "%d out of %d tokens are matched with the GloVe word2vec database" % (len(word_vecs.keys()), len(vocab.keys()))


# Generate random vectors for unknown words
#############################################
print "Generate random vectors for unknown words"

for word in vocab.keys():
	if not word_vecs.has_key(word) and vocab[word] >= MIN_DF:
		word_vecs[word] = np.random.uniform(-0.25, 0.25, K)

print len(word_vecs.keys())


# Get Word Matrix
#############################################

vocab_size = len(word_vecs)
word_idx_map = dict()
W = np.zeros(shape=(vocab_size + 1, K), dtype='float32')
W[0] = np.zeros(K, dtype='float32')

print W[0]
i = 1
print len(word_vecs)

for word in word_vecs:
	W[i,:] = word_vecs[word]
	word_idx_map[word] = i
	i += 1

print word_idx_map['water'] # Index number of water
print len(word_vecs['water']) # Dimension of word vector
print len(W[1, :])


# Turn original text into above indices
# Zero padding here or not?
##########################################################

# Get number of words. For zero padding at the end
max_num_words = max(bills_tok['num_words'])
print "Max: %d, Min: %d" % (max(bills_tok['num_words']), min(bills_tok['num_words']))


# Tokenized text to word matrices
bills_idx = []
for words in bills_tok['main_tokenized']:
	b_idx = []
	b_idx = [word_idx_map.get(word, 0) for word in words]
	bills_idx.append(b_idx)

bills_all = dict(HR = bills_raw['HR'], main_idx = bills_idx, word_idx_map = word_idx_map, W = W)
with open(os.path.join(DATA_PATH, OUT_FILE), 'wb') as f:
	pickle.dump(bills_all, f)