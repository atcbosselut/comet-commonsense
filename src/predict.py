import json
import tqdm
import torch
import logging
import argparse

from src.common import init_model, generate_ending, get_atomic_categories, load_atomic_data_for_generation

logging.basicConfig(
    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)

logger = logging.getLogger(__name__)


def main():
    """
    Generate ATOMIC "then" events given the "if" events
    """
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument("--in_file", default=None, type=str, required=True, help="CSV ATOMIC file")
    parser.add_argument("--out_file", default=None, type=str, required=True, help="jsonl file with input+output events.")
    parser.add_argument("--model_name_or_path", default="models/atomic_pretrained_model", help="Pre-trained COMET model")

    # Optional
    parser.add_argument("--max_length", default=70, type=int, required=False, help="Maximum text length")
    parser.add_argument("--k", default=0, type=int, required=False, help="k for top k sampling")
    parser.add_argument("--p", default=0, type=float, required=False, help="p for nucleus sampling")
    parser.add_argument("--num_beams", default=0, type=int, required=False, help="number of beams in beam search")
    parser.add_argument("--num_samples", default=1, type=int, required=False, help="how many texts to generate")
    parser.add_argument("--device", default="cpu", type=str, help="GPU number or 'cpu'.")
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")
    args = parser.parse_args()
    logger.debug(args)

    # if (args.k == args.p == args.num_beams == 0) or (
    #         sum([1 if i > 0 else 0 for i in [args.k, args.p, args.num_beams]]) > 1):
    #     raise ValueError("Exactly one of p, k, or num_beams should be set to a non-zero value.")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    logger.debug(f"Initializing {args.device}")

    tokenizer, model = init_model(args.model_name_or_path, device=device, do_lower_case=args.do_lower_case)
    categories = get_atomic_categories()

    logger.info(f"Loading ATOMIC examples from {args.in_file}")
    examples = load_atomic_data_for_generation(args.in_file, categories, tokenizer)

    with open(args.out_file, "w", encoding="utf-8") as f_out:
        for input_event, curr_categories in tqdm.tqdm(examples.items()):
            example = {"input": input_event}
            for category, tokens in curr_categories.items():
                example[category] = generate_ending(
                    model, tokenizer, tokens, device, p=args.p, k=args.k,
                    num_beams=args.num_beams, length=args.max_length, num_samples=args.num_samples)

            f_out.write(json.dumps(example) + "\n")


if __name__ == '__main__':
    main()


