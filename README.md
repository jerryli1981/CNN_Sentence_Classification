# CNN_Sentence_Classification

This is a decent implemenation of the paper "Convolutional Neural Networks for Sentence Classification"(EMNLP2014)

Original implementation is here: https://github.com/yoonkim/CNN_sentence.git

To make the code concise and easy understand. I use Lasagne, which is a deep learning framework based on Theano.

---Usage---

1. Follow the command in https://github.com/stanfordnlp/treelstm.git to get Stanford Parser, Stanford POS Tagger. 
2. Run preprocess.sh. Note that you should set your own glove path in this script.
3. Run run.sh

Btw, it is very easy to run the code on GPU, just follow the instructions in Theano website.

