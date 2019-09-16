import sys
import os
import argparse

sys.path.append(os.getcwd())

import pickle

import torch


combine_into_words = {
    'at location': 'AtLocation',
    'capable of': 'CapableOf',
    'causes': 'Causes',
    'causes desire': 'CausesDesire',
    'created by': 'CreatedBy',
    'defined as': 'DefinedAs',
    'desire of': 'DesireOf',
    'desires': 'Desires',
    'has a': 'HasA',
    'has first subevent': 'HasFirstSubevent',
    'has last subevent': 'HasLastSubevent',
    'has pain character': 'HasPainCharacter',
    'has pain intensity': 'HasPainIntensity',
    'has prequisite': 'HasPrerequisite',
    'has property': 'HasProperty',
    'has subevent': 'HasSubevent',
    'inherits from': 'InheritsFrom',
    'instance of': 'InstanceOf',
    'is a': 'IsA',
    'located near': 'LocatedNear',
    'location of action': 'LocationOfAction',
    'made of': 'MadeOf',
    'motivated by goal': 'MotivatedByGoal',
    'not capable of': 'NotCapableOf',
    'not desires': 'NotDesires',
    'not has a': 'NotHasA',
    'not has property': 'NotHasProperty',
    'not is a': 'NotIsA',
    'not made of': 'NotMadeOf',
    'part of': 'PartOf',
    'receives action': 'ReceivesAction',
    'related to': 'RelatedTo',
    'symbol of': 'SymbolOf',
    'used for': 'UsedFor'
}

parser = argparse.ArgumentParser()
parser.add_argument("--gens_file", type=str, default="results/gens/conceptnet-generation/iteration-500-100000/transformer/rel_language-trainsize_100-devversion_12-maxe1_10-maxe2_15/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-init_pt-vSize_40545/exp_generation-seed_123-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-smax_40-sample_greedy-numseq_1-gs_full-es_full/1e-05_adam_64_15500/test.pickle")

args = parser.parse_args()

gens = pickle.load(open(args.gens_file, "rb"))

final_sequences = []

do_beams = False

for idx, gen in enumerate(gens):
    e1 = gen["e1"].strip()
    r = gen["r"]

    if "rel_language" in args.gens_file or r.split(" ")[0] != r:
        r = combine_into_words[r.strip()]
    else:
        r = r.strip("<>")

    if "sequence" in gen:
        sequences = [gen['sequence']]
    else:
        sequences = gen['beams']

    for seq in sequences:
        final_sequences.append("{}\t{}\t{}\t1".format(r, e1, seq))

final_sequences.append("")

print("Saving to: {}".format(args.gens_file.replace("pickle", "txt")))

open(args.gens_file.replace("pickle", "txt"), "w").write("\n".join(final_sequences))
