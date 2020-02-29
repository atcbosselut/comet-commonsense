#!/usr/bin/env bash

COMET_DATA_DIR=$1

# Get ATOMIC data
echo "mkdir -p $COMET_DATA_DIR/data/atomic"
mkdir -p $COMET_DATA_DIR/data/atomic;
echo "wget https://homes.cs.washington.edu/~msap/atomic/data/atomic_data.tgz"
wget https://homes.cs.washington.edu/~msap/atomic/data/atomic_data.tgz;
echo "tar -xvzf atomic_data.tgz -C $COMET_DATA_DIR/data/atomic/"
tar -xvzf atomic_data.tgz -C $COMET_DATA_DIR/data/atomic/;
echo "rm atomic/atomic_data.tgz"
rm atomic_data.tgz;

# Get Pre-trained COMET model
echo "mkdir -p $COMET_DATA_DIR/models/atomic_pretrained_model_openai-gpt"
mkdir -p $COMET_DATA_DIR/models/atomic_pretrained_model_openai-gpt;
echo "gdown https://drive.google.com/uc?id=1z2JkT_fXtmxsQcRmis8KSD4r9YPfrm1A"
gdown https://drive.google.com/uc?id=1z2JkT_fXtmxsQcRmis8KSD4r9YPfrm1A;
echo "tar -xvzf atomic_pretrained_model_openai-gpt.zip"
tar -xvzf  atomic_pretrained_model_openai-gpt.zip;
echo "rm atomic_pretrained_model_openai-gpt.zip"
rm atomic_pretrained_model_openai-gpt.zip;
echo "mv models/gpt/* $COMET_DATA_DIR/models/atomic_pretrained_model_openai-gpt"
mv models/gpt/* $COMET_DATA_DIR/models/atomic_pretrained_model_openai-gpt;

