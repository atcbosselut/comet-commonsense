import os
import sys
import argparse
import torch

sys.path.append(os.getcwd())

import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive


rel_formatting = {
    'AtLocation': "\t\t",
    'CapableOf': "\t\t",
    'Causes': "\t\t",
    'CausesDesire': "\t\t",
    'CreatedBy': "\t\t",
    'DefinedAs': "\t\t",
    'DesireOf': "\t\t",
    'Desires': "\t\t",
    'HasA': "\t\t\t",
    'HasFirstSubevent': "\t",
    'HasLastSubevent': "\t",
    'HasPainCharacter': "\t",
    'HasPainIntensity': "\t",
    'HasPrerequisite': "\t",
    'HasProperty': "\t\t",
    'HasSubevent': "\t\t",
    'InheritsFrom': "\t\t",
    'InstanceOf': "\t\t",
    'IsA': "\t\t\t",
    'LocatedNear': "\t\t",
    'LocationOfAction': "\t",
    'MadeOf': "\t\t",
    'MotivatedByGoal': "\t",
    'NotCapableOf': "\t\t",
    'NotDesires': "\t\t",
    'NotHasA': "\t\t",
    'NotHasProperty': "\t",
    'NotIsA': "\t\t",
    'NotMadeOf': "\t\t",
    'PartOf': "\t\t",
    'ReceivesAction': "\t",
    'RelatedTo': "\t\t",
    'SymbolOf': "\t\t",
    'UsedFor': "\t\t"
}

common_rels = ["AtLocation", "CapableOf", "Causes", "CausesDesire", "CreatedBy", "DefinedAs", "Desires", "HasA", "HasFirstSubevent", "HasLastSubevent", "HasPrerequisite", "HasProperty", "HasSubevent", "IsA", "MadeOf", "MotivatedByGoal", "PartOf", "ReceivesAction", "SymbolOf", "UsedFor"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_file", type=str, default="models/conceptnet-generation/iteration-500-100000/transformer/rel_language-trainsize_100-devversion_12-maxe1_10-maxe2_15/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-init_pt-vSize_40545/exp_generation-seed_123-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-smax_40-sample_greedy-numseq_1-gs_full-es_full-categories_None/1e-05_adam_64_15500.pickle")

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
        input_e1 = "help"
        relation = "help"
        input_e2 = "help"

        while input_e1 is None or input_e1.lower() == "help":
            input_e1 = input("Give an input entity (e.g., go on a hike -- works best if words are lemmatized): ")

            if input_e1 == "help":
                interactive.print_help(opt.dataset)

        while relation.lower() == "help":
            relation = input("Give a relation (type \"help\" for an explanation): ")

            if relation == "help":
                interactive.print_relation_help(opt.dataset)

        while input_e2 is None or input_e2.lower() == "help":
            input_e2 = input("Give an output entity for this input entity and relation (e.g., sleep in tent -- works best if words are lemmatized): ")

            if input_e2 == "help":
                interactive.print_help()

        if relation not in data.conceptnet_data.conceptnet_relations:
            if relation == "common":
                relation = common_rels
            # elif relation == "causal":
                # relation = causal_rels
            else:
                relation = "all"

        outputs = interactive.evaluate_conceptnet_sequence(
            input_e1, model, data_loader, text_encoder, relation, input_e2)

        for key, value in outputs.items():
            print("{} \t {} {} {} \t\t norm: {:.4f} \t tot: {:.4f} \t max: {:.4f} \t step: {}".format(
                input_e1, key, rel_formatting[key], input_e2, value['normalized_loss'],
                value['total_loss'], max(value['step_losses']),
                ["{:.4f}".format(i) for i in value['step_losses']]))
