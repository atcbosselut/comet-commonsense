import tqdm
import logging
import argparse

from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from src.atomic import load_atomic_data
from src.comet_model import PretrainedCometModel

logging.basicConfig(
    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)

logger = logging.getLogger(__name__)


def main():
    """
    Evaluate COMET on ATOMIC
    """
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument("--in_file", default=None, type=str, required=True, help="CSV ATOMIC file")

    # Optional
    parser.add_argument("--model_name_or_path", default="models/atomic_pretrained_model", help="Pre-trained COMET model")
    parser.add_argument("--num_samples", default=10, type=int, required=False, help="how many texts to generate")
    parser.add_argument("--device", default="cpu", type=str, help="GPU number or 'cpu'.")
    parser.add_argument("--max_length", default=25, type=int, required=False, help="Maximum text length")
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")
    args = parser.parse_args()
    logger.debug(args)

    comet_model = PretrainedCometModel(args.model_name_or_path, device=args.device, do_lower_case=args.do_lower_case)

    logger.info(f"Loading ATOMIC examples from {args.in_file}")
    examples = load_atomic_data(args.in_file, comet_model.categories)

    examples_by_category = defaultdict(lambda: defaultdict(list))

    for e1, curr_relations in examples.items():
        for cat, e2s in curr_relations.items():
            examples_by_category[cat][e1.replace("<blank>", "___")] = [f"{e2} <eos>" for e2 in e2s]

    for generation_arg in [{"k": 1}, {"k": 5}, {"k": 10},
                           {"num_beams": 10, "k": 10},
                           {"num_beams": 5, "k": 5},
                           {"num_beams": 2, "k": 2}]:
        logger.info(generation_arg)

        total_bl = {}
        total_count = {}

        for category, curr_examples in examples_by_category.items():
            total_bl[category] = 0
            total_count[category] = 0

            for input_event, list_of_refs in tqdm.tqdm(curr_examples.items()):
                list_of_refs = [ref for ref in list_of_refs if ref != "none"]
                if len(list_of_refs) == 0:
                    continue

                list_of_refs = [comet_model.tokenizer.tokenize(ref) for ref in list_of_refs]

                system_outputs = comet_model.predict(
                    input_event, category, length=args.max_length, num_samples=args.num_samples,
                    return_tokenized=True, **generation_arg)

                bleu_scores = [sentence_bleu(
                    list_of_refs, output, weights=(0.5, 0.5, 0, 0), smoothing_function=SmoothingFunction().method1)
                    for output in system_outputs]

                total_bl[category] += sum(bleu_scores)
                total_count[category] += len(bleu_scores)

            if total_count[category] > 0:
                logger.info(f"{category}: \t {total_bl[category] / total_count[category]}")

    if len(total_bl) > 0:
        total = sum([total_bl[cat] / total_count[cat] for cat in total_bl]) / len(total_bl)
        logger.info(f"Total: \t {total}")


if __name__ == '__main__':
    main()


