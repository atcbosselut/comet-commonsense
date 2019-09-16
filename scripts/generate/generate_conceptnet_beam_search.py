import os
import time
import sys
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--generation_set_size", type=str, default='full', choices=["full", "human"])
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--split", type=str, default="dev")
parser.add_argument("--beam", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--experiment_num", type=str, default="0")
parser.add_argument("--model_name", type=str, default="models/conceptnet-generation/iteration-500-100000/transformer/rel_language-trainsize_100-devversion_12-maxe1_10-maxe2_15/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-npos_1-demb_F-init_pt-vSize_40545/exp_generation-seed_123-es_0-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-trick_0-smax_40-sample_beam-numseq_1-gs_full-es_full-categories_None/1e-05_adam_64_13500.pickle")
parser.add_argument("--gen_len", type=int, default=100)

args = parser.parse_args()
split = args.split

# Generate configuration files depending on experiment being run
utils.generate_config_files("conceptnet", args.experiment_num, eval_mode=True)

# Loads the correct configuration file
config_file = "config/conceptnet/config_{}.json".format(args.experiment_num)

# Read config file to option
config = cfg.read_config(cfg.load_config(config_file))
cfg.device = args.device
eval_opt = cfg.get_eval_parameters(config)

model_stuff = data.load_checkpoint(args.model_name)

opt = model_stuff["opt"]
opt.eval.update(eval_opt)

# Set the random seeds
torch.manual_seed(opt.train.static.seed)
random.seed(opt.train.static.seed)
if config.gpu_mode:
    torch.cuda.manual_seed_all(opt.train.static.seed)

# Where to find the data
splits = ["train", "dev", "test"]

opt.train.dynamic.epoch = 0

print("Loading Data")

if "maxr" not in opt.data.keys():
    opt.data.maxr = 5 if opt.data.rel == "language" else 1
path = "data/conceptnet/processed/generation/{}.pickle".format(
    utils.make_name_string(opt.data))

data_loader = data.make_data_loader(opt)
loaded = data_loader.load_data(path)

data_loader.opt = opt
data_loader.batch_size = opt.train.dynamic.bs

print("Done.")

text_encoder = TextEncoder(config.encoder_path, config.bpe_path)

start_token = "<START>"
end_token = "<END>"

categories = ['AtLocation', 'CapableOf', 'Causes', 'CausesDesire',
              'CreatedBy', 'DefinedAs', 'DesireOf', 'Desires', 'HasA',
              'HasFirstSubevent', 'HasLastSubevent', 'HasPainCharacter',
              'HasPainIntensity', 'HasPrerequisite', 'HasProperty',
              'HasSubevent', 'InheritsFrom', 'InstanceOf', 'IsA',
              'LocatedNear', 'LocationOfAction', 'MadeOf', 'MotivatedByGoal',
              'NotCapableOf', 'NotDesires', 'NotHasA', 'NotHasProperty',
              'NotIsA', 'NotMadeOf', 'PartOf', 'ReceivesAction', 'RelatedTo',
              'SymbolOf', 'UsedFor']

special = [start_token, end_token]
# if opt.data.rel == "relation":
special += ["<{}>".format(cat) for cat in categories]

text_encoder.encoder = data_loader.vocab_encoder
text_encoder.decoder = data_loader.vocab_decoder


context_size_e1 = data_loader.max_e1
context_size_e2 = data_loader.max_e2
context_size_r = data_loader.max_r

n_special = len(special)
n_ctx = context_size_e1 + context_size_e2 + context_size_r
n_vocab = len(text_encoder.encoder) + n_ctx

print(data_loader.__dict__.keys())
opt.net.vSize = n_vocab

print("Building Model")

print(opt.exp)

model = models.make_model(
    opt, n_vocab, n_ctx, 0, load=False, return_acts=True, return_probs=False)

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

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

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


def append_batch(X, beam_toks, mask):
    next_pos = X[:, -1:, 1] + 1
    next_x = torch.cat((beam_toks.unsqueeze(1), next_pos), -1).unsqueeze(1)
    next_mask = torch.cat([mask, torch.ones(X.size(0), 1, device=mask.device)], 1)
    return torch.cat((X, next_x), 1), next_mask


data_loader.reset_offsets(splits=split, shuffle=False)

# Generate for all sequences
b = [tuple(j) for j in data_loader.sequences[split]['positive'][:, :data_loader.max_e1 + data_loader.max_r].tolist()]
total = list(range(len(b)))

args.decoding_strategy = "beam"

kill_mask = torch.ones(args.beam, args.beam).to(device) * 9000
kill_mask[:, 0] = 0

final_sequences = []

end_token = text_encoder.encoder["<END>"]

# Make Eval File name -- can probably comment this if you're doing your
# own file naming convention. I only include this to keep it consistent
eval_file_name = args.model_name.replace("sample_greedy", "sample_{}".format(
    args.decoding_strategy))

eval_file_name = eval_file_name.replace("bs_1", "bs_{}".format(args.beam))
eval_file_name = eval_file_name[:-7] + "/{}.pickle".format(split)
eval_file_name = eval_file_name.replace("models/", "results/gens/")

print("Saving generations to: {}".format(eval_file_name))

with torch.no_grad():
    for idx in tqdm(total):
        sequence_all = {}

        batch, reset = data_loader.sample_batch(split=split, bs=1, idxs=[idx])

        XMB = batch["sequences"][:, :data_loader.max_e1 + data_loader.max_r]
        Ref = batch["sequences"][:, data_loader.max_e1 + data_loader.max_r:]
        MMB = batch["attention_mask"][:, :data_loader.max_e1 + data_loader.max_r]

        init = "".join([text_encoder.decoder[i].replace('</w>', ' ').replace(
                "<blank>", "___ ") for i in XMB[:, :data_loader.max_e1].squeeze().tolist() if i])

        start = data_loader.max_e1
        end = data_loader.max_e1 + data_loader.max_r

        attr = "".join([text_encoder.decoder[i].replace(
            '</w>', ' ') for i in XMB[:, start:end].squeeze(0).tolist() if i]).strip()

        XMB = model_utils.prepare_position_embeddings(
            opt, text_encoder.encoder, XMB.unsqueeze(-1))

        sequence_all["e1"] = init
        sequence_all["r"] = attr
        sequence_all["key"] = batch["key"]

        tokens = []
        beam_losses = []
        # Beam Search
        beam_lls, beam_toks, beam_seqs = None, None, None
        lm_probs = F.log_softmax(lm_model(
            XMB.unsqueeze(1), sequence_mask=MMB), dim=-1)
        dist = lm_probs[:, -1, :].squeeze()
        beam_lls, beam_toks = dist.topk(args.beam)
        beam_losses.append(beam_lls)

        ended = (beam_toks == end_token).float()
        counts = (2 - ended)
        beam_toks = beam_toks.unsqueeze(1)
        beam_seqs = beam_toks.clone()
        XMB = XMB.repeat(args.beam, 1, 1)
        MMB = MMB.repeat(args.beam, 1)
        next_pos = XMB[:, -1:, 1] + 1
        next_x = torch.cat((beam_toks, next_pos), -1).unsqueeze(1)
        XMB = torch.cat((XMB, next_x), 1)
        MMB = torch.cat([MMB, torch.ones(XMB.size(0), 1, device=MMB.device)], 1)

        for _ in range(args.gen_len):

            # Compute distribution for current beam
            lm_probs = F.log_softmax(lm_model(
                XMB.unsqueeze(1), sequence_mask=MMB), dim=-1)
            dist = lm_probs[:, -1, :].squeeze()

            # get hypothesis tokens for distribution
            hyp_beam_lls, hyp_beam_toks = dist.topk(args.beam)

            # Compute masks and expand beam
            expanded_ended = ended.unsqueeze(1).repeat(1, args.beam)
            hypothesis_mask = expanded_ended * kill_mask + (1 - expanded_ended)
            current_beam_lls = beam_lls.unsqueeze(1).repeat(
                1, args.beam).view(args.beam**2)

            # Compute losses of hypotheses, masking those that have ended
            hyp_beam_lls = (hyp_beam_lls.view(args.beam**2) *
                            hypothesis_mask.view(-1)) + current_beam_lls

            # Get normalizer for sequences
            temp_counts = counts.unsqueeze(1).repeat(1, args.beam).view(
                args.beam ** 2)

            # Select best beams with lowest aggregate loss
            beam_lls, top_beam_idxs = (hyp_beam_lls / temp_counts).topk(args.beam)

            # Update placements in beam based on selecetion
            beam_losses = [i.index_select(0, top_beam_idxs // args.beam)
                           for i in beam_losses]
            ended = ended.index_select(0, top_beam_idxs // args.beam)
            counts = temp_counts.index_select(0, top_beam_idxs)

            # Save beam losses
            beam_losses.append(beam_lls * counts)

            # Update beam tokens
            ended_mask = (1 - ended).long()
            end_replacement = (end_token * ended).long()
            next_toks = hyp_beam_toks.view(-1)[top_beam_idxs]
            beam_toks = next_toks * ended_mask + end_replacement

            # Update ended and counts
            ended = ended + (beam_toks == end_token).float() * (1 - ended)
            counts = counts + (1 - ended)

            # Update beam sequences
            beam_seqs = beam_seqs.t().repeat(args.beam, 1).t().contiguous().view(args.beam**2, -1)[top_beam_idxs]
            beam_seqs = torch.cat((beam_seqs, beam_toks.unsqueeze(1)), dim=1)

            # I have no idea what's going on but Ari's on point with it
            XMB = XMB.transpose(0, 1).transpose(1, 2).repeat(
                args.beam, 1, 1).transpose(2, 1).transpose(
                1, 0).contiguous().view(
                args.beam**2, XMB.size(1), XMB.size(2))[top_beam_idxs]

            XMB, MMB = append_batch(XMB, beam_toks, MMB)

            if (beam_toks == end_token).sum().item() == args.beam or _ == context_size_e2 - 1:
                break

        for tok in beam_seqs[0]:
            tokens.append(text_encoder.decoder[tok.item()].replace('</w>', ' ').replace('\n', ''))

        beams = []

        for beam in beam_seqs:
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

