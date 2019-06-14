mkdir data/conceptnet

cd data/conceptnet

wget https://ttic.uchicago.edu/~kgimpel/comsense_resources/train100k.txt.gz
wget https://ttic.uchicago.edu/~kgimpel/comsense_resources/dev1.txt.gz
wget https://ttic.uchicago.edu/~kgimpel/comsense_resources/dev2.txt.gz
wget https://ttic.uchicago.edu/~kgimpel/comsense_resources/test.txt.gz

gunzip train100k.txt.gz
gunzip dev1.txt.gz
gunzip dev2.txt.gz
gunzip test.txt.gz

cd ..