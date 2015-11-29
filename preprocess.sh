#!/bin/bash

CLASSPATH="lib:lib/stanford-parser/stanford-parser.jar:lib/stanford-parser/stanford-parser-3.5.2-models.jar"
javac -cp $CLASSPATH lib/*.java
python preprocess.py --glove /Users/peng/Develops/NLP-Tools/glove.840B.300d.txt

