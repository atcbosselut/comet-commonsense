import os
import sys
import argparse
import demo_bilinear

train_file = "data/conceptnet/train100k.txt.gz"

parser = argparse.ArgumentParser()
parser.add_argument("--gens_name", type=str, default="results/gens/conceptnet-generation/iteration-500-100000/transformer/rel_language-trainsize_100-devversion_12-maxe1_10-maxe2_15/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-init_pt-vSize_40545/exp_generation-seed_123-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-smax_40-sample_greedy-numseq_1-gs_full-es_full/1e-05_adam_64_15500/test.txt")
parser.add_argument("--thresh", type=float, default=0.5)

args = parser.parse_args()

# print(gens_file[0])
results = demo_bilinear.run(args.gens_name, flip_r_e1=False)
new_results = {"0": [j for (i, j) in results if i[3] == "0"],
               "1": [j for (i, j) in results if i[3] == "1"]}

print("Total")
num_examples = 1.0 * len(results)
accuracy = (len([i for i in new_results["1"] if i >= args.thresh]) +
            len([i for i in new_results["0"] if i < args.thresh])) / num_examples
print("Accuracy @ {}: {}".format(args.thresh, accuracy))

