import numpy as np
import os

def iterate_minibatches_(inputs, batchsize, shuffle=False):

    if shuffle:
        indices = np.arange(len(inputs[0]))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs[0]) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield ( input[excerpt] for input in inputs )

def loadWord2VecMap(word2vec_path):
    import cPickle as pickle
    
    with open(word2vec_path,'r') as fid:
        return pickle.load(fid)

def read_sequence_dataset(dataset_dir, dataset_name, maxlen=60):


    a_s = os.path.join(dataset_dir, dataset_name+"/a.toks")
    labs = os.path.join(dataset_dir, dataset_name+"/label.txt") 

    data_size = len([line.rstrip('\n') for line in open(a_s)])

    Y_scores_pred = np.zeros((data_size, 6), dtype=np.float32)    
    Y_scores = np.zeros((data_size), dtype=np.float32) 
    labels = []

    X = np.zeros((data_size, maxlen), dtype=np.int16)

    X_mask = np.zeros((data_size, maxlen), dtype=np.int16)

    from collections import defaultdict
    words = defaultdict(int)

    vocab_path = os.path.join(dataset_dir, 'vocab-cased.txt')

    with open(vocab_path, 'r') as f:
        for tok in f:
            words[tok.rstrip('\n')] += 1

    vocab = {}
    vocab["<UNK>"] = 0
    for word, idx in zip(words.iterkeys(), xrange(1, len(words)+1)):
        vocab[word] = idx

    with open(a_s, "rb") as f1, open(labs, 'rb') as f4:
                        
        for i, (a, ent) in enumerate(zip(f1,f4)):

            a = a.rstrip('\n')
            label = ent.rstrip('\n')

            labels.append(label)

            toks_a = a.split()

            for j in range(maxlen):
                if j < maxlen - len(toks_a):
                    X[i,j] = vocab["<UNK>"]
                    X_mask[i, j] = 0
                else:
                    X[i, j] = vocab[toks_a[j-maxlen+len(toks_a)]]
                    X_mask[i, j] = 1
      
    Y_labels = np.zeros((len(labels), 2))
    for i in range(len(labels)):
        Y_labels[i, labels[i]] = 1.

    return X, X_mask, Y_labels