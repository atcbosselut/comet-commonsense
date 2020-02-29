#!/usr/bin/env bash

DATA_DIR=$1

# Get ATOMIC data
echo "mkdir -p ${DATA_DIR}/data/atomic"
mkdir -p ${DATA_DIR}/data/atomic;
echo "wget https://homes.cs.washington.edu/~msap/atomic/data/atomic_data.tgz -O ${DATA_DIR}/data/atomic/atomic_data.tgz"
wget https://homes.cs.washington.edu/~msap/atomic/data/atomic_data.tgz;
echo "tar -xvzf atomic_data.tgz -C ${DATA_DIR}a/data/atomic/"
tar -xvzf atomic_data.tgz -C ${DATA_DIR}/data/atomic/;
echo "rm atomic/atomic_data.tgz"
rm atomic_data.tgz;

# Get Pre-trained COMET model
echo "mkdir -p ${DATA_DIR}/models/"
mkdir -p ${DATA_DIR}/models/;
echo "gdown https://drive.google.com/uc?id=1z2JkT_fXtmxsQcRmis8KSD4r9YPfrm1A"
gdown https://drive.google.com/uc?id=1z2JkT_fXtmxsQcRmis8KSD4r9YPfrm1A;
echo "unzip atomic_pretrained_model.zip -d ${DATA_DIR}/models/"
unzip atomic_pretrained_model.zip -d ${DATA_DIR}/models/;
echo "rm atomic_pretrained_model_openai-gpt.zip"
rm atomic_pretrained_model_openai-gpt.zip;
echo "mv ${DATA_DIR}/models/atomic_pretrained_model_openai-gpt ${DATA_DIR}/models/"
mv ${DATA_DIR}/models/atomic_pretrained_model_openai-gpt ${DATA_DIR}/models/;
