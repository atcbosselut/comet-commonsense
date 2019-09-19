import os
import time
import sys

sys.path.append(os.getcwd())

from nltk import bleu
from nltk.translate.bleu_score import SmoothingFunction
from tqdm import tqdm

import pandas
import json
import pickle

import src.data.data as data
from utils.utils import DD
import utils.utils as utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=2)
parser.add_argument("--gens_file", type=str, default="results/gens/atomic-generation/iteration-500-50000/transformer/categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant-maxe1_17-maxe2_35-maxr_1/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-tembs_F-drel_F-de1_F-de2_F-dpos_T-init_pt-vSize_40542/exp_generation-seed_123-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_10-smax_40-sample_beam-numseq_1-gs_full-es_1000-categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant/6.25e-05_adam_64_22000/{}.gens")

args = parser.parse_args()

def get_data_params(gens_file):
    data_str = gens_file.split("/")[5]
    data_objs = data_str.split("-")
    data_params = {}
    for case in data_objs:
        if case.split("_")[1].isdigit():
            data_params[case.split("_")[0]] = int(case.split("_")[1])
        elif "#" in case.split("_")[1]:
            data_params[case.split("_")[0]] = case.split("_")[1].split("#")
        else:
            data_params[case.split("_")[0]] = case.split("_")[1]
    return data_params

gens_file = args.gens_file
split = gens_file.split("/")[-1].split(".")[0]
n = args.n

def flatten(outer):
    return [el for key in outer for el in key]

opt = DD()
opt.data = DD()
opt.dataset = "atomic"
opt.exp = "generation"

data_params = get_data_params(gens_file)

categories = data_params["categories"]#sorted(["oReact", "oEffect", "oWant", "xAttr", "xEffect", "xIntent", "xNeed", "xReact", "xWant"])

opt.data.categories = data_params["categories"]

if "maxe1" in data_params:
    opt.data.maxe1 = data_params["maxe1"]
    opt.data.maxe2 = data_params["maxe2"]
    opt.data.maxr = data_params["maxr"]

path = "data/atomic/processed/generation/{}.pickle".format(
    utils.make_name_string(opt.data))
data_loader = data.make_data_loader(opt, categories)
loaded = data_loader.load_data(path)

refs = {}

for i in range(data_loader.sequences[split]["total"].size(0)):
    sequence = data_loader.sequences[split]["total"][i]
    tmp = sequence[:data_loader.max_event + 1]
    init = "".join([data_loader.vocab_decoder[i].replace('</w>', ' ').replace("<blank>", "___ ") for i in tmp[:-1].squeeze().tolist() if i])
    attr = data_loader.vocab_decoder[tmp[-1].item()].strip("<>")
    Ref = sequence[data_loader.max_event + 1:]
    ref = "".join([data_loader.vocab_decoder[i].replace('</w>', ' ').replace("<blank>", "___ ") for i in Ref.squeeze().tolist() if i and i != data_loader.vocab_encoder["<END>"]])

    refs.setdefault(init, {})
    refs[init][attr] = refs[init].get(attr, []) + [ref]

def get_event(event):
    if "<" in event and ">" in event:
        return event[:event.index("<")]
    else:
        return event

gens = pickle.load(open(gens_file, "rb"))

# Set score
weights = [1/n] * n

def score(hyp, refs):
    return bleu(refs, hyp, weights=weights, smoothing_function=SmoothingFunction().method1)

# Compute BLEU
total_bl = {}
total_count = {}

for category in categories:

    total_bl[category] = 0
    total_count[category] = 0

    temp_gens = [k for k in gens if k['effect_type'] == category]

    for gen in tqdm(temp_gens):
        event = gen["event"]
        list_of_gens = gen['beams']
        list_of_refs = refs[get_event(event)][gen['effect_type']]

        clean_list_of_refs = [[j for j in i.split() if j != '<unk>' and j != "<END>"] for i in list_of_refs]
        clean_list_of_gens = [[j for j in i.split() if j != '<unk>' and j != "<END>"] for i in list_of_gens]

        if sum([i == ["none"] for i in clean_list_of_refs]) / len(clean_list_of_refs) > 1/3:
            continue

        example_bl = []

        for clean_gen in clean_list_of_gens:

            example_bl.append(score(clean_gen, clean_list_of_refs))

        total_bl[category] += sum(example_bl)
        total_count[category] += len(example_bl)

    print("{}: \t {}".format(category, total_bl[category] / total_count[category]))

total = sum([total_bl[cat] / total_count[cat] for cat in total_bl]) / len(total_bl)

print("Total: \t {}".format(total))

write_obj = {
    "total": total,
    "category": {}}
write_obj["category"] = {
    "raw": total_bl,
    "counts": total_count,
    "bleu": {category: total_bl[category] / total_count[category] for category in total_bl}}

write_name = gens_file.replace(".gens", ".bleu.json")

print("Saving to: {}".format(write_name))
with open(write_name, "w") as f:
    json.dump(write_obj, f)
