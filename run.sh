#!/bin/bash

export THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32

python main_lasagne.py 