import os
import sys
import pickle
import argparse

sys.path.append(os.getcwd())

import src.data.conceptnet as cdata

combine_into_words = {j:i for i, j in cdata.split_into_words.items()}

parser = argparse.ArgumentParser()
parser.add_argument("--gens_name", type=str, default="results/gens/conceptnet-generation/iteration-500-100000/transformer/rel_language-trainsize_100-devversion_12-maxe1_10-maxe2_15/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-init_pt-vSize_40545/exp_generation-seed_123-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-smax_40-sample_greedy-numseq_1-gs_full-es_full/1e-05_adam_64_15500/test.txt")
parser.add_argument("--training_set_file", type=str, default="data/concepnet/train100k.txt")

args = parser.parse_args()

gens = pickle.load(open(args.gens_name, "rb"))
training_gens = [i.split("\t")[:3] for i in open(args.training_set_file, "r").read().split("\n") if i]

evaluation_rels = []
evaluation_e2s = []

do_beams = len(gens[0]['beams']) > 1

for idx, example in enumerate(gens):
    e1 = example["e1"]

    if example["r"].strip("<>") not in cdata.split_into_words:
        r = combine_into_words[example["r"]]
    else:
        r = example["r"].strip("<>")
#
    if do_beams:
        examples_sequences = example["beams"]
    else:
        examples_sequences = [example['sequence']]

    for seq in examples_sequences:
        evaluation_rels.append((r.strip(), e1.strip(), seq.strip()))
        evaluation_e2s.append(seq.strip())

train_rels = set([tuple([j.strip() for j in i]) for i in training_gens])
train_e2s = set([i[2].strip() for i in training_gens])

print("% new o:   {}".format(len([i for i in evaluation_e2s if i not in train_e2s]) / len(evaluation_rels)))
print("% new sro: {}".format(len([i for i in evaluation_rels if i not in train_rels]) / len(evaluation_rels)))
