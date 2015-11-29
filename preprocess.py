import os
import glob
import random
import re

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def dependency_parse(filepath, cp='', tokenize=True):
    print('\nDependency parsing ' + filepath)
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + '.toks')
    parentpath = os.path.join(dirpath, filepre + '.parents')
    relpath =  os.path.join(dirpath, filepre + '.rels')
    tokenize_flag = '-tokenize - ' if tokenize else ''
    cmd = ('java -cp %s DependencyParse -tokpath %s -parentpath %s -relpath %s %s < %s'
        % (cp, tokpath, parentpath, relpath, tokenize_flag, filepath))
    os.system(cmd)


def build_vocab(filepaths, dst_path, lowercase=True):
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab |= set(line.split())
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')

def split(filepath, dst_dir):
    with open(filepath) as datafile, \
         open(os.path.join(dst_dir, 'a.txt'), 'w') as afile, \
         open(os.path.join(dst_dir, 'label.txt'),'w') as labelfile:
            datafile.readline()
            for line in datafile:
                a, label = line.strip().split('\t')
                afile.write(a+'\n')
                labelfile.write(label+'\n')

def parse(dirpath, cp=''):
    dependency_parse(os.path.join(dirpath, 'a.txt'), cp=cp, tokenize=True)


def build_word2Vector(glove_path, sick_dir, vocab_name):

    print "building word2vec"
    from collections import defaultdict
    import numpy as np
    words = defaultdict(int)

    vocab_path = os.path.join(sick_dir, 'vocab-cased.txt')

    with open(vocab_path, 'r') as f:
        for tok in f:
            words[tok.rstrip('\n')] += 1

    vocab = {}
    vocab["<UNK>"] = 0
    for word, idx in zip(words.iterkeys(), xrange(1, len(words)+1)):
        vocab[word] = idx

    print "word size", len(words)
    print "vocab size", len(vocab)


    word_embedding_matrix = np.zeros(shape=(300, len(vocab)))  

    
    import gzip
    wordSet = defaultdict(int)

    with open(glove_path, "rb") as f:
        for line in f:
           toks = line.split(' ')
           word = toks[0]
           if word in vocab:
                wordIdx = vocab[word]
                word_embedding_matrix[:,wordIdx] = np.fromiter(toks[1:], dtype='float32')
                wordSet[word] +=1
    
    count = 0   
    for word in vocab:
        if word not in wordSet:
            wordIdx = vocab[word]
            count += 1
            word_embedding_matrix[:,wordIdx] = np.random.uniform(-0.05,0.05, 300) 
    
    print "Number of words not in glove ", count
    import cPickle as pickle
    with open(os.path.join(sick_dir, 'word2vec.bin'),'w') as fid:
        pickle.dump(word_embedding_matrix,fid)

def generate_datasets(data_dir, pos, neg):

    li = []
    with open(pos, 'r') as f1, open(neg, 'r') as f2:
         for pos_s, neg_s in zip(f1, f2):
            pos_s = clean_str(pos_s)
            neg_s = clean_str(neg_s)
            pos_s = pos_s.rstrip('\n')
            neg_s = neg_s.rstrip('\n')
            pos_s += "\t1"
            neg_s += "\t0"
            li.append(pos_s)
            li.append(neg_s)

    random.shuffle(li)
    len_li = len(li)

    train_path = os.path.join(data_dir, "train.txt")
    dev_path = os.path.join(data_dir, "dev.txt")
    test_path = os.path.join(data_dir, "test.txt")

    with open(train_path, 'w') as train, open(dev_path, 'w') as dev, open(test_path , 'w') as test:
        for sent in li[len_li/2:]:
            test.write(sent+"\n")

        for sent in li[:len_li/2]:
            if random.randint(1, 10) == 5:
                dev.write(sent+"\n")
            else:
                train.write(sent+"\n")



if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing dataset')
    print('=' * 80)

    import argparse
    parser = argparse.ArgumentParser(description="Usage")
    parser.add_argument("--glove",dest="glove",type=str,default=None)
    args = parser.parse_args()
    glove_path = args.glove

    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    lib_dir = os.path.join(base_dir, 'lib')
    train_dir = os.path.join(data_dir, 'train')
    dev_dir = os.path.join(data_dir, 'dev')
    test_dir = os.path.join(data_dir, 'test')
    make_dirs([train_dir, dev_dir, test_dir])

    # java classpath for calling Stanford parser
    classpath = ':'.join([
        lib_dir,
        os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.5.2-models.jar')])

    generate_datasets(data_dir, "rt-polarity.pos", "rt-polarity.neg")

    # split into separate files
    split(os.path.join(data_dir, 'train.txt'), train_dir)
    split(os.path.join(data_dir, 'dev.txt'), dev_dir)
    split(os.path.join(data_dir, 'test.txt'), test_dir)

    # parse sentences

    parse(train_dir, cp=classpath)
    parse(dev_dir, cp=classpath)
    parse(test_dir, cp=classpath)

    # get vocabulary
    build_vocab(
        glob.glob(os.path.join(data_dir, '*/*.toks')),
        os.path.join(data_dir, 'vocab.txt'))
    build_vocab(
        glob.glob(os.path.join(data_dir, '*/*.toks')),
        os.path.join(data_dir, 'vocab-cased.txt'),
        lowercase=False)

    vocab_path = os.path.join(data_dir, 'vocab-cased.txt')
    build_word2Vector(glove_path, data_dir, 'vocab-cased.txt')
