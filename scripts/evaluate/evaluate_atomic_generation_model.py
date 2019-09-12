import os
import time
import sys

sys.path.append(os.getcwd())

import src.data.config as cfg

cfg.device = 0

import torch
import argparse

import src.data.data as data
import src.data.atomic as adata
import src.data.config as cfg
import src.models.models as models
import src.evaluate.atomic_evaluate as evaluate
import utils.utils as utils
from src.data.utils import TextEncoder

torch.cuda.set_device(cfg.device)

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_num", type=str, default="0")
parser.add_argument("--split", type=str, default="dev")
parser.add_argument("--model_name", type=str, default="models/atomic-generation/iteration-500-50000/transformer/categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-init_pt-vSize_40542/exp_generation-seed_123-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-smax_40-sample_greedy-numseq_1-gs_1000-es_1000-categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant/6.25e-05_adam_64_22000.pickle")

args = parser.parse_args()
split = args.split

# eval_mode = True means changes are taken from config/atomic/eval_changes.json
utils.generate_config_files("atomic", args.experiment_num, eval_mode=True)

# Loads the correct configuration file
config_file = "config/atomic/config_{}.json".format(args.experiment_num)

print(config_file)

# Read config file to option
config = cfg.read_config(cfg.load_config(config_file))
cfg.device = config.gpu_index
eval_opt = cfg.get_eval_parameters(config)

# Batch multiple models
model_file = data.load_checkpoint(args.model_name)
opt = model_file["opt"]

opt.eval.update(eval_opt)

print("Loading Data")

# Do multiple sets of categories:
# compute individual perplexity of categories in addition to total perplexity
if len(opt.data.categories) == 1:
    set_of_categories = [opt.data.categories]
else:
    set_of_categories = [opt.data.categories] + [[i] for i in opt.data.categories]

print(set_of_categories)

# Iterate over sets of categories to compute perplexities for
for eval_categories in set_of_categories:
    print(eval_categories)
    opt.eval.categories = eval_categories

    results_name = "{}/{}.{}".format(utils.make_name(
        opt, prefix="results/{}/".format("losses"),
        is_dir=True, eval_=True), split, "pickle")
    print("Will save {} losses to {}".format(split, results_name))

    path = "data/atomic/processed/generation/{}.pickle".format(
        utils.make_name_string(opt.data).replace(
            "kr_{}".format(opt.data.get("kr", 1)), "kr_1"))
    data_loader = data.make_data_loader(opt, opt.data.categories)
    loaded = data_loader.load_data(path)

    data_loader.batch_size = opt.train.dynamic.bs

    print("Done.")

    text_encoder = TextEncoder(config.encoder_path, config.bpe_path)

    # Set special tokens
    formatted_categories = ["<{}>".format(cat) for cat in opt.data.categories]

    special = [data.start_token, data.end_token]
    special += formatted_categories
    special += [data.blank_token]

    # Load vocab encoder and decoder from pre-initialized data_loader
    text_encoder.encoder = data_loader.vocab_encoder
    text_encoder.decoder = data_loader.vocab_decoder

    # Get component segmentation of sequences
    # context_size_event = maximum size of an event description
    # context_size_effect = maximum size of an event effect/intent/etc.
    context_size_event = data_loader.max_event
    context_size_effect = data_loader.max_effect

    n_special = len(special)
    n_ctx = context_size_event + context_size_effect
    n_vocab = len(text_encoder.encoder) + n_ctx

    opt.net.vSize = n_vocab

    # Prune data from data loader depending on the evaluation set
    if not all([i in opt.eval.categories for i in opt.data.categories]):
        print("Pruning Data")
        print("Original number of evaluation sequences: {}".format(
            len(data_loader.sequences[split]["total"])))

        adata.prune_data_for_evaluation(
            data_loader,
            ["<{}>".format(cat) for cat in opt.eval.categories],
            split)

        print("Pruned number of evaluation sequences for subset: {}".format(
            len(data_loader.sequences[split]["total"])))

    print("Building Model")

    model = models.make_model(
        opt, n_vocab, n_ctx, n_special, load=False)

    print("Loading Weights")
    models.load_state_dict(model, model_file["state_dict"])

    print("Done Loading Weights")

    model.eval()

    # Initialize variable for # of examples to cycle through
    data.set_max_sizes(data_loader, force_split=split)

    evaluator = evaluate.make_evaluator(opt, model, data_loader)
    evaluator.batch_variables["split"] = split
    model.cuda(cfg.device)

    loss = evaluator.epoch(opt, model, data_loader, split)

    data.save_eval_file(opt, loss, "losses", split=split)

    loss_str = []
    loss_str.append("Per Token   Loss:       {}".format(loss["total_micro"]))
    loss_str.append("Per Token   Perplexity: {}".format(loss["ppl_micro"]))
    loss_str.append("Per Example Loss:       {}".format(loss["total_macro"]))
    loss_str.append("Per Example Perplexity: {}".format(loss["ppl_macro"]))
    loss_str = "\n".join(loss_str)

    print(loss_str)

    data.save_eval_file(opt, loss_str, "losses", split=split, ext="txt")
