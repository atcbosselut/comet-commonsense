wget https://storage.googleapis.com/ai2-mosaic/public/atomic/v1.0/atomic_data.tgz
mkdir -p data/atomic
mv atomic_data.tgz data/atomic

tar -xvzf data/atomic/atomic_data.tgz -C data/atomic
rm data/atomic/atomic_data.tgz
