#!/usr/bin/env bash

# Get ATOMIC data
echo "mkdir -p ~/.comet-data/data/atomic"
mkdir -p ~/.comet-data/data/atomic;
echo "wget https://homes.cs.washington.edu/~msap/atomic/data/atomic_data.tgz -O ~/.comet-data/data/atomic/atomic_data.tgz"
wget https://homes.cs.washington.edu/~msap/atomic/data/atomic_data.tgz -O ~/.comet-data/data/atomic/atomic_data.tgz;
echo "tar -xvzf ~/.comet-data/data/atomic/atomic_data.tgz"
tar -xvzf ~/.comet-data/data/atomic/atomic_data.tgz;
echo "rm ~/.comet-data/data/atomic/atomic_data.tgz"
rm ~/.comet-data/data/atomic/atomic_data.tgz;

# Get Pre-trained COMET model
echo "gdown https://drive.google.com/uc?id=1aIIoTz9m28yyW1ygg6OG-JkAVLUyuIRh"
gdown https://drive.google.com/uc?id=1aIIoTz9m28yyW1ygg6OG-JkAVLUyuIRh;
echo "mkdir -p ~/.comet-data/models/"
mkdir -p ~/.comet-data/models/;
echo "mv atomic_pretrained_models.zip ~/.comet-data/models/"
mv atomic_pretrained_models.zip ~/.comet-data/models/;
echo "unzip ~/.comet-data/models/atomic_pretrained_models.zip"
unzip ~/.comet-data/models/atomic_pretrained_models.zip;
