import os
import time
import sys

sys.path.append(os.getcwd())
import torch

import src.train.atomic_train as train
import src.models.models as models
import src.data.data as data
import utils.utils as utils
import src.train.utils as train_utils
import src.data.config as cfg

from src.data.utils import TextEncoder
from src.train.opt import OpenAIAdam

import src.models.utils as model_utils
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--generation_set_size", type=str, default='full', choices=["full", "human"])
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--split", type=str, default="dev")
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--experiment_num", type=str, default="0")
parser.add_argument("--model_name", type=str, default="models/atomic-generation/iteration-500-50000/transformer/categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant-maxe1_17-maxe2_35-maxr_1/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-init_pt-vSize_40542/exp_generation-seed_123-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-smax_40-sample_greedy-numseq_1-gs_1000-es_1000-categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant/6.25e-05_adam_64_22000.pickle")
parser.add_argument("--gen_len", type=int, default=100)

args = parser.parse_args()
split = args.split

# Generate configuration files depending on experiment being run
utils.generate_config_files("atomic", args.experiment_num, eval_mode=True)

# Loads the correct configuration file
config_file = "config/atomic/config_{}.json".format(args.experiment_num)

# Read config file to option
config = cfg.read_config(cfg.load_config(config_file))
cfg.device = config.gpu_index
eval_opt = cfg.get_eval_parameters(config)


model_stuff = data.load_checkpoint(args.model_name)

opt = model_stuff["opt"]
opt.eval.update(eval_opt)

opt.train.dynamic.epoch = 0

print("Loading Data")

categories = opt.data.categories

path = "data/atomic/processed/generation/{}.pickle".format(
    utils.make_name_string(opt.data))

data_loader = data.make_data_loader(opt, categories)
loaded = data_loader.load_data(path)

data_loader.batch_size = opt.train.dynamic.bs

print("Done.")

text_encoder = TextEncoder(config.encoder_path, config.bpe_path)

special = [data.start_token, data.end_token]
special += ["<{}>".format(cat) for cat in categories]

special += [data.blank_token]

text_encoder.encoder = data_loader.vocab_encoder
text_encoder.decoder = data_loader.vocab_decoder

context_size_event = data_loader.max_event
context_size_effect = data_loader.max_effect

n_special = len(special)
n_ctx = context_size_event + context_size_effect
n_vocab = len(text_encoder.encoder) + n_ctx

print(data_loader.__dict__.keys())
opt.net.vSize = n_vocab

print("Building Model")

print(opt.exp)

model = models.make_model(
    opt, n_vocab, n_ctx, 0, load=False, return_acts=False, return_probs=True)

models.load_state_dict(model, model_stuff["state_dict"])

if config.gpu_mode:
    print("Pushing to GPU: {}".format(config.gpu_index))
    cfg.device = config.gpu_index
    cfg.do_gpu = True
    torch.cuda.set_device(cfg.device)
    model.cuda(cfg.device)
    print("Done.")

model.eval()

device = cfg.device
model.to(device)

lm_model = model


def make_batch(X):
    X = np.array(X)
    assert X.ndim in [1, 2]
    if X.ndim == 1:
        X = np.expand_dims(X, axis=0)
    pos_enc = np.arange(n_vocab + n_special, n_vocab + n_special + X.shape[-1])
    pos_enc = np.expand_dims(pos_enc, axis=0)
    batch = np.stack([X, pos_enc], axis=-1)
    batch = torch.tensor(batch, dtype=torch.long).to(device)
    return batch


def append_batch(X, next_idx, mask):
    next_pos = X[:, -1:, 1] + 1
    next_x = torch.cat((next_idx, next_pos), -1).unsqueeze(1)
    next_mask = torch.cat([mask, torch.ones(X.size(0), 1, device=mask.device)], 1)
    return torch.cat((X, next_x), 1), next_mask


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

data_loader.reset_offsets(splits=args.split, shuffle=False)

# Get unique contexts
# Generate for all sequences
if args.generation_set_size == "full":
    b = [tuple(j) for j in data_loader.sequences[split]['total'][:, :data_loader.max_event + 1].tolist()]
    total = []
    set_total = set()
    for i, sequence in enumerate(b):
        if sequence not in set_total:
            total.append(i)
            set_total.add(sequence)
elif args.generation_set_size == "human":  # Generate for events in human evaluation only
    human_events = open("data/atomic/{}-human-eval-events.txt".format(split), "r").read().split("\n")
    found = []
    total = []
    for i, j in enumerate(data_loader.data[split]["total"]):
        if j[0] in human_events and (j[0], j[1]) not in found:
            found.append((j[0], j[1]))
            total.append(i)
else:
    total = list(range(int(args.generation_set_size)))

final_sequences = []

end_token = text_encoder.encoder["<END>"]

eval_file_name = args.model_name.replace("sample_greedy", "sample_topk")
eval_file_name = eval_file_name.replace("bs_1", "k_{}".format(args.k))
eval_file_name = eval_file_name.replace("gs_1000", "gs_{}".format(args.generation_set_size))
eval_file_name = eval_file_name[:-7] + "/{}.gens".format(args.split)
eval_file_name = eval_file_name.replace("models/", "results/gens/")

print("Saving generations to: {}".format(eval_file_name))

with torch.no_grad():
    for idx in tqdm(total):
        sequence_all = {}

        batch, reset = data_loader.sample_batch(split=args.split, bs=1, idxs=[idx])

        tmp = batch["sequences"][:, :context_size_event + 1]

        XMB = batch["sequences"][:, :context_size_event + 1]
        Ref = batch["sequences"][:, context_size_event + 1:]
        MMB = batch["attention_mask"][:, :context_size_event + 1]

        init = "".join([text_encoder.decoder[i].replace('</w>', ' ').replace(
                "<blank>", "___ ") for i in XMB[:, :-1].squeeze().tolist() if i])
        attr = text_encoder.decoder[XMB[:, -1].item()].strip("<>")

        XMB = model_utils.prepare_position_embeddings(
            opt, text_encoder.encoder, XMB.unsqueeze(-1))

        sequence_all["event"] = init
        sequence_all["effect_type"] = attr

        lm_probs = lm_model(XMB.unsqueeze(1), sequence_mask=MMB)
        dist = lm_probs[:, -1, :].squeeze()

        values, indices = lm_probs[:, -1, :].topk(args.k)
        seqs = indices.t().clone()

        ended = (seqs == end_token).float()
        XMB = XMB.repeat(args.k, 1, 1)
        MMB = MMB.repeat(args.k, 1)
        next_pos = XMB[:, -1:, 1] + 1
        next_x = torch.cat((indices.view(args.k, -1), next_pos), -1).unsqueeze(1)
        XMB = torch.cat((XMB, next_x), 1)
        MMB = torch.cat([MMB, torch.ones(XMB.size(0), 1, device=MMB.device)], 1)

        # Sample from top k

        steps = 0

        while ended.sum().item() != args.k and steps != context_size_effect:
            lm_probs = lm_model(XMB.unsqueeze(1), sequence_mask=MMB)
            dist = lm_probs[:, -1, :].squeeze()

            # Sample from top k
            values, indices = lm_probs[:, -1, :].topk(args.k)
            choice = torch.multinomial(values, 1)
            next_idx = indices.gather(-1, choice)

            ended = ended + (next_idx == end_token).float() * (1 - ended)

            next_idx = next_idx * (1 - ended).long() + ended.long() * end_token

            seqs = torch.cat([seqs, next_idx], 1)

            steps += 1

            XMB, MMB = append_batch(XMB, next_idx, MMB)

        beams = []

        for beam in seqs:
            beams.append(" ".join("".join(
                [text_encoder.decoder[tok.item()].replace(
                    '</w>', ' ').replace('\n', '')
                 for tok in beam if tok != end_token]).split()))

        sequence_all['beams'] = beams
        final_sequences.append(sequence_all)

import pickle

utils.mkpath("/".join(eval_file_name.split("/")[:-1]))

with open(eval_file_name, "wb") as f:
    pickle.dump(final_sequences, f)

