import os
import tqdm
import logging
import argparse

from nltk import bleu
from collections import defaultdict
from nltk.translate.bleu_score import SmoothingFunction

from comet2.atomic import load_atomic_data
from comet2.comet_model import PretrainedCometModel, BASE_DIR, MODEL_DIR

logging.basicConfig(
    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(BASE_DIR, "models")


def main():
    """
    Evaluate COMET on ATOMIC
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_file", type=str, help="CSV ATOMIC file",
                        default=os.path.join(DATA_DIR, "v4_atomic_dev.csv"))
    parser.add_argument("--model_name_or_path",
                        default=os.path.join(MODEL_DIR, "atomic_pretrained_model_openai-gpt"),
                        help="Pre-trained COMET model")
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--num_beams", default=5, type=int, required=False, help="how many texts to generate")
    parser.add_argument("--device", default="cpu", type=str, help="GPU number or 'cpu'.")
    parser.add_argument("--max_length", default=25, type=int, required=False, help="Maximum text length")
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")
    args = parser.parse_args()
    logger.debug(args)

    comet_model = PretrainedCometModel(args.model_name_or_path, device=args.device, do_lower_case=args.do_lower_case)

    smoothing = SmoothingFunction().method1
    weights = [1 / args.n] * args.n

    logger.info(f"Loading ATOMIC examples from {args.in_file}")
    examples = load_atomic_data(args.in_file, comet_model.categories)

    examples_by_category = defaultdict(lambda: defaultdict(list))

    # Group by category
    for e1, curr_relations in examples.items():
        for cat, e2s in curr_relations.items():
            examples_by_category[cat][e1.replace("<blank>", "___")] = e2s

    total_bl = {}
    total_count = {}

    for category, curr_examples in examples_by_category.items():
        total_bl[category] = 0
        total_count[category] = 0

        for input_event, refs in tqdm.tqdm(curr_examples.items()):

            # Skip empty references
            refs = [comet_model.tokenizer.tokenize(ref) for ref in refs]
            refs = [t[:t.index("<eos>")] if "<eos>" in t else t for t in refs]
            refs = [[w for w in t if w != "<unk>"] for t in refs]
            refs = [tuple([w.replace("</w>", "") for w in ref]) for ref in refs]

            if len(refs) == 0 or sum([i == [("none",)] for i in refs]) / len(refs) > 1/3:
                continue

            sys_outputs = comet_model.predict(
                input_event, category, length=args.max_length, return_tokenized=True, num_beams=args.num_beams)

            bleu_scores = [100.0 * bleu(refs, out, weights=weights, smoothing_function=smoothing) for out in sys_outputs]
            total_bl[category] += sum(bleu_scores)
            total_count[category] += len(bleu_scores)

        if total_count[category] > 0:
            logger.info(f"{category}: \t {total_bl[category] / total_count[category]}")

    if len(total_bl) > 0:
        total = sum([total_bl[cat] / total_count[cat] for cat in total_bl]) / len(total_bl)
        logger.info(f"Total: \t {total}")


if __name__ == '__main__':
    main()


