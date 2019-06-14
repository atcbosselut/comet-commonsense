import os
import sys
import argparse
import torch

sys.path.append(os.getcwd())

import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_file", type=str, default="models/conceptnet-generation/iteration-500-100000/transformer/rel_language-trainsize_100-devversion_12-maxe1_10-maxe2_15/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-init_pt-vSize_40545/exp_generation-seed_123-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-smax_40-sample_greedy-numseq_1-gs_full-es_full-categories_None/1e-05_adam_64_15500.pickle")
    parser.add_argument("--sampling_algorithm", type=str, default="help")

    args = parser.parse_args()

    opt, state_dict = interactive.load_model_file(args.model_file)

    data_loader, text_encoder = interactive.load_data("conceptnet", opt)

    n_ctx = data_loader.max_e1 + data_loader.max_e2 + data_loader.max_r
    n_vocab = len(text_encoder.encoder) + n_ctx

    model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)

    if args.device != "cpu":
        cfg.device = int(args.device)
        cfg.do_gpu = True
        torch.cuda.set_device(cfg.device)
        model.cuda(cfg.device)
    else:
        cfg.device = "cpu"


    while True:
        input_event = "help"
        relation = "help"
        sampling_algorithm = args.sampling_algorithm

        while input_event is None or input_event.lower() == "help":
            input_event = input("Give an input entity (e.g., go on a hike -- works best if words are lemmatized): ")

            if input_event == "help":
                interactive.print_help(opt.dataset)

        while relation.lower() == "help":
            relation = input("Give a relation (type \"help\" for an explanation): ")

            if relation == "help":
                interactive.print_relation_help(opt.dataset)

        while sampling_algorithm.lower() == "help":
            sampling_algorithm = input("Give a sampling algorithm (type \"help\" for an explanation): ")

            if sampling_algorithm == "help":
                interactive.print_sampling_help()

        sampler = interactive.set_sampler(opt, sampling_algorithm, data_loader)

        if relation not in data.conceptnet_data.conceptnet_relations:
            relation = "all"

        outputs = interactive.get_conceptnet_sequence(
            input_event, model, sampler, data_loader, text_encoder, relation)

