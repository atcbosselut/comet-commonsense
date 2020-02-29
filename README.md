This repository contains a new version of COMET trained on ATOMIC. 

For the original version see: [atcbosselut/comet-commonsense](https://github.com/atcbosselut/comet-commonsense).

### Changes from previous version

1. Variable length input

### Installation

Install the repository. This will also download the ATOMIC dataset and the pre-trained COMET model:


```
pip install git+https://github.com/vered1986/comet-commonsense.git
```


### Using a pre-trained model

The installation comes with a pre-trained model based on GPT. 

```
>>> from comet2.comet_model import PretrainedCometModel

>>> comet_model = PretrainedCometModel(device=1)

>>> comet_model.predict("PersonX asked PersonY for an example for the demo", "xWant")
['to have y respond to personx']

>>> comet_model.predict("PersonX just woke up", "xEffect")
['gets out of bed']
```

The performance of the pre-trained model is:

* **Micro perplexity**: 11.87 (original model: 11.14)
* **BLEU-2**: 14.43 (original model: 15.10)

You can also specify a different model path `model_name_or_path` when you create `PretrainedCometModel`.


### Training

Run `python -m comet2.train` with the following arguments:

```
usage: train.py [-h] [--train_file TRAIN_FILE] --out_dir OUT_DIR
                [--adam_epsilon ADAM_EPSILON] [--device DEVICE] [--do_eval]
                [--do_lower_case] [--do_train]
                [--eval_batch_size EVAL_BATCH_SIZE]
                [--eval_data_file EVAL_DATA_FILE] [--eval_during_train]
                [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                [--learning_rate LEARNING_RATE]
                [--logging_steps LOGGING_STEPS]
                [--max_input_length MAX_INPUT_LENGTH]
                [--max_output_length MAX_OUTPUT_LENGTH]
                [--max_grad_norm MAX_GRAD_NORM] [--max_steps MAX_STEPS]
                [--model_name_or_path MODEL_NAME_OR_PATH]
                [--model_type MODEL_TYPE]
                [--num_train_epochs NUM_TRAIN_EPOCHS] [--overwrite_cache]
                [--overwrite_out_dir] [--save_steps SAVE_STEPS]
                [--save_total_limit SAVE_TOTAL_LIMIT] [--seed SEED]
                [--train_batch_size TRAIN_BATCH_SIZE]
                [--warmup_steps WARMUP_STEPS] [--weight_decay WEIGHT_DECAY]

optional arguments:
  -h, --help            show this help message and exit
  --train_file TRAIN_FILE
                        The input training CSV file.
  --out_dir OUT_DIR     Out directory for checkpoints.
  --adam_epsilon ADAM_EPSILON
                        Epsilon for Adam optimizer.
  --device DEVICE       GPU number or 'cpu'.
  --do_eval             Whether to run eval on the dev set.
  --do_lower_case       Set this flag if you are using an uncased model.
  --do_train            Whether to run training.
  --eval_batch_size EVAL_BATCH_SIZE
                        Batch size for evaluation.
  --eval_data_file EVAL_DATA_FILE
                        Validation file
  --eval_during_train   Evaluate at each train logging step.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Steps before backward pass.
  --learning_rate LEARNING_RATE
                        The initial learning rate for Adam.
  --logging_steps LOGGING_STEPS
                        Log every X updates steps.
  --max_input_length MAX_INPUT_LENGTH
                        Maximum input event length in words.
  --max_output_length MAX_OUTPUT_LENGTH
                        Maximum output event length in words.
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm.
  --max_steps MAX_STEPS
                        If > 0: total number of training steps to perform.
  --model_name_or_path MODEL_NAME_OR_PATH
                        LM checkpoint for initialization.
  --model_type MODEL_TYPE
                        The LM architecture to be fine-tuned.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Number of training epochs to perform.
  --overwrite_cache     Overwrite the cached data.
  --overwrite_out_dir   Overwrite the output directory.
  --save_steps SAVE_STEPS
                        Save checkpoint every X updates steps.
  --save_total_limit SAVE_TOTAL_LIMIT
                        Maximum number of checkpoints to keep
  --seed SEED           Random seed for initialization.
  --train_batch_size TRAIN_BATCH_SIZE
                        Batch size for training.
  --warmup_steps WARMUP_STEPS
                        Linear warmup over warmup_steps.
  --weight_decay WEIGHT_DECAY
                        Weight decay if we apply some.
```

### Evaluation

The training script can be used to evaluate with perplexity. 
Use the `--do_eval` flag and set `--eval_data_file` to the validation set. 


To get BLEU scores, run `python -m comet2.evaluate` with the following arguments:

```
usage: evaluate.py [-h] [--in_file IN_FILE]
                   [--model_name_or_path MODEL_NAME_OR_PATH]
                   [--num_samples NUM_SAMPLES] [--device DEVICE]
                   [--max_length MAX_LENGTH] [--do_lower_case]

optional arguments:
  -h, --help            show this help message and exit
  --in_file IN_FILE     CSV ATOMIC file
  --model_name_or_path MODEL_NAME_OR_PATH
                        Pre-trained COMET model
  --num_samples NUM_SAMPLES
                        how many texts to generate
  --device DEVICE       GPU number or 'cpu'.
```

### Generation

To run an interactive script for single predictions: `python -m comet2.interactive`

```
usage: interactive.py [-h] [--model_name_or_path MODEL_NAME_OR_PATH]
                      [--sampling_algorithm SAMPLING_ALGORITHM]
                      [--device DEVICE] [--max_length MAX_LENGTH]
                      [--do_lower_case]

optional arguments:
  -h, --help            show this help message and exit
  --model_name_or_path MODEL_NAME_OR_PATH
                        Pre-trained COMET model
  --sampling_algorithm SAMPLING_ALGORITHM
  --device DEVICE       GPU number or 'cpu'.
  --max_length MAX_LENGTH
                        Maximum text length
  --do_lower_case       Set this flag if you are using an uncased model.
```

To generate predictions for a dataset, run `python -m comet2.predict` with the following arguments:

```
usage: predict.py [-h] --out_file OUT_FILE [--in_file IN_FILE]
                  [--model_name_or_path MODEL_NAME_OR_PATH]
                  [--max_length MAX_LENGTH] [--k K] [--p P]
                  [--num_beams NUM_BEAMS] [--num_samples NUM_SAMPLES]
                  [--device DEVICE] [--do_lower_case]

optional arguments:
  -h, --help            show this help message and exit
  --out_file OUT_FILE   jsonl file with input+output events.
  --in_file IN_FILE     CSV ATOMIC file
  --model_name_or_path MODEL_NAME_OR_PATH
                        Pre-trained COMET model
  --max_length MAX_LENGTH
                        Maximum text length
  --k K                 k for top k sampling
  --p P                 p for nucleus sampling
  --num_beams NUM_BEAMS
                        number of beams in beam search
  --num_samples NUM_SAMPLES
                        how many texts to generate
  --device DEVICE       GPU number or 'cpu'.
  --do_lower_case       Set this flag if you are using an uncased model.
```


### References 

Please cite this repository using the following reference:

```
@inproceedings{Bosselut2019COMETCT,
  title={COMET: Commonsense Transformers for Automatic Knowledge Graph Construction},
  author={Antoine Bosselut and Hannah Rashkin and Maarten Sap and Chaitanya Malaviya and Asli Çelikyilmaz and Yejin Choi},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2019}
}
```
