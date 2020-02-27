#!/usr/bin/env bash

# Get ATOMIC data
echo "mkdir -p ~/.comet-data/data/atomic"
mkdir -p ~/.comet-data/data/atomic;
echo "wget https://homes.cs.washington.edu/~msap/atomic/data/atomic_data.tgz -O ~/.comet-data/data/atomic/atomic_data.tgz"
wget https://homes.cs.washington.edu/~msap/atomic/data/atomic_data.tgz;
echo "tar -xvzf atomic_data.tgz -C ~/.comet-data/data/atomic/"
tar -xvzf atomic_data.tgz -C ~/.comet-data/data/atomic/;
echo "rm atomic/atomic_data.tgz"
rm atomic_data.tgz;

# Get Pre-trained COMET model
echo "mkdir -p ~/.comet-data/models/"
mkdir -p ~/.comet-data/models/;
echo "gdown https://drive.google.com/uc?id=1aIIoTz9m28yyW1ygg6OG-JkAVLUyuIRh"
gdown https://drive.google.com/uc?id=1aIIoTz9m28yyW1ygg6OG-JkAVLUyuIRh;
echo "unzip atomic_pretrained_model.zip -d ~/.comet-data/models/"
unzip atomic_pretrained_model.zip -d ~/.comet-data/models/;
echo "rm atomic_pretrained_model.zip"
rm atomic_pretrained_model.zip;

# Configuration
echo "cp -r ../config ~/.comet-data/"
cp -r ../config ~/.comet-data/;
