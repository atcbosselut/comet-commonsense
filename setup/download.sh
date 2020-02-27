#!/usr/bin/env bash

# Get ATOMIC data
wget https://homes.cs.washington.edu/~msap/atomic/data/atomic_data.tgz
mkdir -p data/atomic
mv atomic_data.tgz data/atomic
tar -xvzf data/atomic/atomic_data.tgz -C data/atomic
rm data/atomic/atomic_data.tgz

# Get Pre-trained COMET model
gdown gdown https://drive.google.com/uc?id=1aIIoTz9m28yyW1ygg6OG-JkAVLUyuIRh
mkdir -p models/
unzip atomic_pretrained_models.zip
mv atomic_pretrained_models models/
