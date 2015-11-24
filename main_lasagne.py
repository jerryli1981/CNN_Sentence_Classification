import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

from scipy.stats import pearsonr

sys.path.insert(0, os.path.abspath('../Lasagne'))

from lasagne.layers import InputLayer, LSTMLayer, NonlinearityLayer, SliceLayer, FlattenLayer, EmbeddingLayer,\
    ElemwiseMergeLayer, ReshapeLayer, get_output, get_all_params, get_output_shape, DropoutLayer,\
    DenseLayer,ElemwiseSumLayer,Conv2DLayer, Conv1DLayer, CustomRecurrentLayer, AbsSubLayer,\
    ConcatLayer, Pool1DLayer, FeaturePoolLayer,count_params,MaxPool2DLayer,MaxPool1DLayer

from lasagne.regularization import regularize_layer_params_weighted, l2, l1,regularize_layer_params,\
                                    regularize_network_params
from lasagne.nonlinearities import tanh, sigmoid, softmax, rectify
from lasagne.objectives import categorical_crossentropy, squared_error, categorical_accuracy, binary_crossentropy,\
                                binary_accuracy
from lasagne.updates import sgd, adagrad, adadelta, nesterov_momentum, rmsprop, adam
from lasagne.init import GlorotUniform

from utils import read_sequence_dataset, iterate_minibatches_,loadWord2VecMap


def build_network_2dconv(args, input_var, target_var, wordEmbeddings, maxlen=60):

    
    print("Building model with 2D Convolution")

    vocab_size = wordEmbeddings.shape[1]
    wordDim = wordEmbeddings.shape[0]

    num_filters = 100
    stride = 1 

    #CNN_sentence config
    filter_size=(3, wordDim)
    pool_size=(maxlen-3+1,1)

    input = InputLayer((None, maxlen),input_var=input_var)
    batchsize, seqlen = input.input_var.shape
    emb = EmbeddingLayer(input, input_size=vocab_size, output_size=wordDim, W=wordEmbeddings.T)
    emb.params[emb.W].remove('trainable') #(batchsize, maxlen, wordDim)

    reshape = ReshapeLayer(emb, (batchsize, 1, maxlen, wordDim))

    conv2d = Conv2DLayer(reshape, num_filters=num_filters, filter_size=(filter_size), stride=stride, 
        nonlinearity=rectify,W=GlorotUniform()) #(None, 100, 34, 1)
    maxpool = MaxPool2DLayer(conv2d, pool_size=pool_size) #(None, 100, 1, 1) 
  
    forward = FlattenLayer(maxpool) #(None, 100) #(None, 50400)
 
    hid = DenseLayer(forward, num_units=args.hiddenDim, nonlinearity=sigmoid)

    network = DenseLayer(hid, num_units=2, nonlinearity=softmax)

    prediction = get_output(network)
    
    loss = T.mean(binary_crossentropy(prediction,target_var))
    lambda_val = 0.5 * 1e-4

    layers = {conv2d:lambda_val, hid:lambda_val, network:lambda_val} 
    penalty = regularize_layer_params_weighted(layers, l2)
    loss = loss + penalty


    params = get_all_params(network, trainable=True)

    if args.optimizer == "sgd":
        updates = sgd(loss, params, learning_rate=args.step)
    elif args.optimizer == "adagrad":
        updates = adagrad(loss, params, learning_rate=args.step)
    elif args.optimizer == "adadelta":
        updates = adadelta(loss, params, learning_rate=args.step)
    elif args.optimizer == "nesterov":
        updates = nesterov_momentum(loss, params, learning_rate=args.step)
    elif args.optimizer == "rms":
        updates = rmsprop(loss, params, learning_rate=args.step)
    elif args.optimizer == "adam":
        updates = adam(loss, params, learning_rate=args.step)
    else:
        raise "Need set optimizer correctly"
 

    test_prediction = get_output(network, deterministic=True)
    test_loss = T.mean(binary_crossentropy(test_prediction,target_var))

    train_fn = theano.function([input_var, target_var], 
        loss, updates=updates, allow_input_downcast=True)

    test_acc = T.mean(binary_accuracy(test_prediction, target_var))
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc], allow_input_downcast=True)

    return train_fn, val_fn

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Usage")

    parser.add_argument("--minibatch",dest="minibatch",type=int,default=30)
    parser.add_argument("--optimizer",dest="optimizer",type=str,default="adagrad")
    parser.add_argument("--epochs",dest="epochs",type=int,default=2)
    parser.add_argument("--step",dest="step",type=float,default=0.01)
    parser.add_argument("--hiddenDim",dest="hiddenDim",type=int,default=50)
    args = parser.parse_args()


    # Load the dataset
    print("Loading data...")
    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    sick_dir = os.path.join(data_dir, 'sick')

    X_train, X_mask_train, Y_labels_train = read_sequence_dataset(sick_dir, "train")
    X_dev, X_mask_dev, Y_labels_dev = read_sequence_dataset(sick_dir, "dev")
    X_test, X_mask_test, Y_labels_test= read_sequence_dataset(sick_dir, "test")

    input_var = T.imatrix('inputs')
    target_var = T.fmatrix('targets')

    wordEmbeddings = loadWord2VecMap(os.path.join(sick_dir, 'word2vec.bin'))
    wordEmbeddings = wordEmbeddings.astype(np.float32)

    train_fn, val_fn = build_network_2dconv(args, input_var, target_var, wordEmbeddings)

    print("Starting training...")
    best_val_acc = 0
    best_val_pearson = 0
    for epoch in range(args.epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches_((X_train, X_mask_train, Y_labels_train), args.minibatch, shuffle=True):

            inputs, _, labels= batch
            train_err += train_fn(inputs, labels)
            train_batches += 1
 
        val_err = 0
        val_acc = 0
        val_batches = 0
        val_pearson = 0

        for batch in iterate_minibatches_((X_dev, X_mask_dev, Y_labels_dev), len(X_dev), shuffle=False):

            inputs, inputs_mask, labels= batch

            err, acc = val_fn(inputs, labels)
            val_acc += acc

            val_err += err
            
            val_batches += 1

            
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, args.epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

        val_score = val_acc / val_batches * 100
        print("  validation accuracy:\t\t{:.2f} %".format(val_score))
        if best_val_acc < val_score:
            best_val_acc = val_score

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_pearson = 0
    test_batches = 0
    for batch in iterate_minibatches_((X_test, X_mask_test, Y_labels_test), len(X_test), shuffle=False):

        inputs, inputs_mask, labels= batch

        err, acc = val_fn(inputs, labels)
        test_acc += acc
        test_err += err
        
        test_batches += 1


    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))


    print("  Best validate accuracy:\t\t{:.2f} %".format(best_val_acc))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))
