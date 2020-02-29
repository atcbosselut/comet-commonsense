import os
import json
import tqdm
import logging
import argparse

from comet2.atomic import load_atomic_data
from comet2.comet_model import PretrainedCometModel, MODEL_DIR, BASE_DIR

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(BASE_DIR, "models")


def main():
    """
    Generate ATOMIC "then" events given the "if" events
    """
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument("--out_file", default=None, type=str, required=True, help="jsonl file with input+output events.")

    # Optional
    parser.add_argument("--in_file", type=str, help="CSV ATOMIC file",
                        default=os.path.join(DATA_DIR, "v4_atomic_dev.csv"))
    parser.add_argument("--model_name_or_path",
                        default=os.path.join(MODEL_DIR, "atomic_pretrained_model_openai-gpt"),
                        help="Pre-trained COMET model")

    parser.add_argument("--max_length", default=70, type=int, required=False, help="Maximum text length")
    parser.add_argument("--k", default=0, type=int, required=False, help="k for top k sampling")
    parser.add_argument("--p", default=0, type=float, required=False, help="p for nucleus sampling")
    parser.add_argument("--num_beams", default=0, type=int, required=False, help="number of beams in beam search")
    parser.add_argument("--num_samples", default=1, type=int, required=False, help="how many texts to generate")
    parser.add_argument("--device", default="cpu", type=str, help="GPU number or 'cpu'.")
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")
    args = parser.parse_args()
    logger.debug(args)

    if (args.k == args.p == args.num_beams == 0) or (
            sum([1 if i > 0 else 0 for i in [args.k, args.p, args.num_beams]]) > 1):
        raise ValueError("Exactly one of p, k, or num_beams should be set to a non-zero value.")

    comet_model = PretrainedCometModel(args.model_name_or_path, device=args.device, do_lower_case=args.do_lower_case)

    logger.info(f"Loading ATOMIC examples from {args.in_file}")
    examples = load_atomic_data(args.in_file, comet_model.categories)
    examples = {e1: list(e1_relations.keys()) for e1, e1_relations in examples.items()}

    with open(args.out_file, "w", encoding="utf-8") as f_out:
        for input_event, curr_categories in tqdm.tqdm(examples.items()):
            example = {"input": input_event}
            for category in curr_categories:
                example[category] = comet_model.predict(
                    input_event, category, p=args.p, k=args.k, num_beams=args.num_beams, length=args.max_length,
                    num_samples=args.num_samples)

            f_out.write(json.dumps(example) + "\n")


if __name__ == '__main__':
    main()


