import tqdm
import torch
import logging
import argparse

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from src.common import init_model, generate_ending, get_atomic_categories, load_atomic_data_for_eval

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
    parser.add_argument("--num_samples", default=10, type=int, required=False, help="how many texts to generate")
    parser.add_argument("--device", default="cpu", type=str, help="GPU number or 'cpu'.")
    parser.add_argument("--max_length", default=25, type=int, required=False, help="Maximum text length")
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")
    args = parser.parse_args()
    logger.debug(args)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    logger.debug(f"Initializing {args.device}")

    tokenizer, model = init_model(args.model_name_or_path, device=device, do_lower_case=args.do_lower_case)
    categories = get_atomic_categories()

    logger.info(f"Loading ATOMIC examples from {args.in_file}")
    examples = load_atomic_data_for_eval(args.in_file, categories, tokenizer)

    for generation_arg in [{"k": 1}, {"k": 5}, {"k": 10},
                           {"num_beams": 10, "k": 10},
                           {"num_beams": 5, "k": 5},
                           {"num_beams": 2, "k": 2}]:
        logger.info(generation_arg)

        total_bl = {}
        total_count = {}

        for category, curr_examples in examples.items():
            total_bl[category] = 0
            total_count[category] = 0

            for input_event, (input_tokens, list_of_refs) in tqdm.tqdm(curr_examples.items()):
                list_of_refs = [ref for ref in list_of_refs if ref != "none"]
                if len(list_of_refs) == 0:
                    continue

                list_of_refs = [tokenizer.tokenize(ref) for ref in list_of_refs]

                system_outputs = generate_ending(
                    model, tokenizer, input_tokens, device, length=args.max_length, num_samples=args.num_samples,
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


