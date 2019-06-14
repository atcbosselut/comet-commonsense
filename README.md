To run a generation experiment (either conceptnet or atomic), follow these instructions:


<h1>First Steps</h1>

First clone, the repo:

```
git clone https://github.com/atcbosselut/comet-commonsense.git
```

Then run the setup scripts to acquire the pretrained model files from OpenAI, as well as the ATOMIC and ConceptNet datasets

```
bash scripts/setup/get_atomic_data.sh
bash scripts/setup/get_conceptnet_data.sh
bash scripts/setup/get_model_files.sh
```

Then install dependencies (assuming you already have Python 3.6 and Pytorch >= 1.0:

```
pip install tensorflow
pip install ftfy==5.1
conda install -c conda-forge spacy
python -m spacy download en
pip install tensorboardX
pip install tqdm
pip install pandas
pip install ipython
```


<h1> Making the Data Loaders </h1>

Run the following scripts to pre-initialize a data loader for ATOMIC or ConceptNet:

```
python scripts/data/make_atomic_data_loader.py
python scripts/data/make_conceptnet_data_loader.py
```

For the ATOMIC KG, if you'd like to make a data loader for only a subset of the relation types, comment out any relations in lines 17-25. 
For ConceptNet if you'd like to map the relations to natural language analogues, set ```opt.data.rel = "language"``` in line 26. If you want to initialize unpretrained relation tokens, set ```opt.data.rel = "relation"```

<h1> Setting the ATOMIC configuration files </h1>

Open ```config/atomic/changes.json``` and set which categories you want to train, as well as any other details you find important. Check ```src/data/config.py``` for a description of different options. Variables you may want to change: batch_size, learning_rate, categories. See ```config/default.json``` and ```config/atomic/default.json``` for default settings of some of these variables.

<h1> Setting the ConceptNet configuration files </h1>

Open ```config/conceptnet/changes.json``` and set any changes to the degault configuration that you may want to vary in this experiment. Check ```src/data/config.py``` for a description of different options. Variables you may want to change: batch_size, learning_rate, etc. See ```config/default.json``` and ```config/conceptnet/default.json``` for default settings of some of these variables.

<h1> Running the ATOMIC experiment </h1>

<h3> Training </h3>
For whichever experiment # you set in ```config/atomic/changes.json``` (e.g., 0, 1, 2, etc.), run:

```
python src/main.py --experiment_type atomic --experiment_num #
```

<h3> Evaluation </h3>

Once you've trained a model, run the evaluation script:

```
python scripts/evaluate/evaluate_atomic_generation_model.py --split $DATASET_SPLIT --model_name /path/to/model/file
```

<h3> Generation </h3>

Once you've trained a model, run the generation script for the type of decoding you'd like to do:

```
python scripts/generate/generate_atomic_beam_search.py --beam 10 --split $DATASET_SPLIT --model_name /path/to/model/file
python scripts/generate/generate_atomic_greedy.py --split $DATASET_SPLIT --model_name /path/to/model/file
python scripts/generate/generate_atomic_topk.py --k 10 --split $DATASET_SPLIT --model_name /path/to/model/file
```

<h1> Running the ConceptNet experiment </h1>

<h3> Training </h3>

For whichever experiment # you set in ```config/conceptnet/changes.json``` (e.g., 0, 1, 2, etc.), run:

```
python src/main.py --experiment_type conceptnet --experiment_num #
```

Development and Test set tuples are automatically evaluated and generated with greedy decoding during training

<h3> Generation </h3>

If you want to generate with a larger beam size, run the generation script

```
python scripts/generate/generate_conceptnet_beam_search.py --beam 10 --split $DATASET_SPLIT --model_name /path/to/model/file
```

<h1> Playing Around in Interactive Mode </h1>

First, download the pretrained models from the following link:

```
https://drive.google.com/open?id=1FccEsYPUHnjzmX-Y5vjCBeyRt1pLo8FB
```

Then untar the file:

```
tar -xvzf pretrained_models.tar.gz
```

Then run the following script to interactively generate arbitrary ATOMIC event effects:

```
python scripts/interactive/atomic_single_example.py --model_file pretrained_models/atomic_pretrained_model.pickle
```

Or run the following script to interactively generate arbitrary ConceptNet tuples:

```
python scripts/interactive/conceptnet_single_example.py --model_file pretrained_models/conceptnet_pretrained_model.pickle
```
