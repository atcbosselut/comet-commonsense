import tqdm
import torch
import logging
import argparse

from nltk import bleu
from nltk.translate.bleu_score import SmoothingFunction

from src.fine_tune.common import init_model, generate_ending, get_atomic_categories, load_atomic_data_for_eval

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
    parser.add_argument("--model_name_or_path", default="openai-gpt", type=str, help="LM checkpoint.")
    parser.add_argument("--model_type", default="openai-gpt", type=str, help="The LM architecture.")
    parser.add_argument("--k", default=0, type=int, required=False, help="k for top k sampling")
    parser.add_argument("--p", default=0, type=float, required=False, help="p for nucleus sampling")
    parser.add_argument("--num_beams", default=0, type=int, required=False, help="number of beams in beam search")
    parser.add_argument("--num_samples", default=1, type=int, required=False, help="how many texts to generate")
    parser.add_argument("--device", default="cpu", type=str, help="GPU number or 'cpu'.")
    parser.add_argument("--max_input_length", default=50, type=int, help="Maximum input event length in words.")
    parser.add_argument("--max_output_length", default=50, type=int, help="Maximum output event length in words.")
    args = parser.parse_args()
    logger.debug(args)

    if (args.k == args.p == args.num_beams == 0) or (
            sum([1 if i > 0 else 0 for i in [args.k, args.p, args.num_beams]]) > 1):
        raise ValueError("Exactly one of p, k, or num_beams should be set to a non-zero value.")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    logger.debug(f"Initializing {args.device}")

    model, tokenizer = init_model(args.model_name_or_path, args.model_type, device)
    categories = get_atomic_categories()

    logger.info(f"Loading ATOMIC examples from {args.in_file}")
    examples = load_atomic_data_for_eval(
        args.in_file, categories, tokenizer, args.max_input_length, args.max_output_length)

    # Compute BLEU
    total_bl = {}
    total_count = {}

    for category, curr_examples in examples.items():
        total_bl[category] = 0
        total_count[category] = 0

        for input_event, (list_of_refs, input_tokens, list_of_output_tokens) in tqdm.tqdm(curr_examples.items()):
            system_outputs = generate_ending(
                model, tokenizer, input_tokens, device, p=args.p, k=args.k,
                num_beams=args.num_beams, length=args.max_output_length, num_samples=args.num_samples)

            example_bl = [bleu(output, list_of_refs, smoothing_function=SmoothingFunction().method1)
                          for output in system_outputs]

            total_bl[category] += sum(example_bl)
            total_count[category] += len(example_bl)

        if total_count[category] > 0:
            logger.info(f"{category}: \t {total_bl[category] / total_count[category]}")

    if len(total_bl) > 0:
        total = sum([total_bl[cat] / total_count[cat] for cat in total_bl]) / len(total_bl)
        logger.info(f"Total: \t {total}")


if __name__ == '__main__':
    main()


