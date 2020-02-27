#!/usr/bin/env bash

# Get ATOMIC data
wget https://homes.cs.washington.edu/~msap/atomic/data/atomic_data.tgz
mkdir -p ~/.comet-data/data/atomic
mv atomic_data.tgz ~/.comet-data/data/atomic
tar -xvzf data/atomic/atomic_data.tgz -C ~/.comet-data/data/atomic
rm ~/.comet-data/data/atomic/atomic_data.tgz

# Get Pre-trained COMET model
gdown gdown https://drive.google.com/uc?id=1aIIoTz9m28yyW1ygg6OG-JkAVLUyuIRh
mkdir -p ~/.comet-data/models/
unzip atomic_pretrained_models.zip
mv atomic_pretrained_models ~/.comet-data/models/
