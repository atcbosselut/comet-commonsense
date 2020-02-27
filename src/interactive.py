import torch
import logging
import argparse

from src.common import init_model, generate_ending, get_atomic_categories

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
    parser.add_argument("--model_name_or_path", default="models/atomic_pretrained_model", help="Pre-trained COMET model")

    # Optional
    parser.add_argument("--sampling_algorithm", type=str, default="topk-1")
    parser.add_argument("--device", default="cpu", type=str, help="GPU number or 'cpu'.")
    parser.add_argument("--max_length", default=70, type=int, required=False, help="Maximum text length")
    args = parser.parse_args()
    logger.debug(args)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    logger.debug(f"Initializing {args.device}")

    tokenizer, model = init_model(args.model_name_or_path, device=device, do_lower_case=True)
    categories = set(get_atomic_categories())

    while True:
        input_event = "help"
        category = "help"
        sampling_algorithm = args.sampling_algorithm

        while input_event is None or input_event.lower() == "help":
            input_event = input("Give an event (e.g., PersonX went to the mall), or quit: ").lower()

            if input_event == "quit":
                return

            if input_event == "help":
                print("""Provide a seed event such as "PersonX goes to the mall".\n
                         Don't include names, instead replacing them with PersonX, PersonY, etc.\n
                         The event should always have PersonX included.""")

        while category.lower() == "help":
            category = input("Give an effect type (type \"help\" for an explanation): ")

            if category == "help":
                print("""Enter a possible effect type from the following effect types:\n
                         all - compute the output for all effect types 
                         {{oEffect, oReact, oWant, xAttr, xEffect, xIntent, xNeed, xReact, xWant}}\n
                         oEffect - generate the effect of the event on participants other than PersonX\n
                         oReact - generate the reactions of participants other than PersonX to the event\n
                         oEffect - generate what participants other than PersonX may want after the event\n""")

        while sampling_algorithm.lower() == "help":
            sampling_algorithm = input("Give a sampling algorithm (type \"help\" for an explanation): ")

            if sampling_algorithm == "help":
                print("""Provide a sampling algorithm to produce the sequence with from the following:\n\n
                greedy\n 
                beam-# where # is the beam size\n
                topk-# where # is k\n
                topp-# where # is p\n\n""")

        sampler = get_sampler(sampling_algorithm)

        if category not in categories:
            category = "all"

        fields = {"input_event": input_event, "effect_type": category, "results": {}}
        categories_to_get = categories if category == "all" else [category]

        for cat in categories_to_get:
            tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{input_event} <{cat}>"))
            fields["results"][cat] = generate_ending(model, tokenizer, tokens, device, length=args.max_length, **sampler)

        print_atomic_sequence(fields)


def print_atomic_sequence(sequence_object):
    input_event = sequence_object["input_event"]
    category = sequence_object["effect_type"]

    print("Input Event:   {}".format(input_event))
    print("Target Effect: {}".format(category))
    print("")
    print("Candidate Sequences:")
    for cat, results in sequence_object["results"].items():
        print(cat)
        print("=======")
        for result in results:
            print(result)
        print('')
    print("")
    print("====================================================")
    print("")


def get_sampler(sampling_algorithm):
    if "beam" in sampling_algorithm:
        num_beams = int(sampling_algorithm.split("-")[1])
        return {"num_beams": num_beams, "num_samples": num_beams, "k": num_beams}
    elif "topk" in sampling_algorithm:
        k = int(sampling_algorithm.split("-")[1])
        return {"k": k}
    elif "topp" in sampling_algorithm:
        p = float(sampling_algorithm.split("-")[1])
        return {"p": p}
    # Default - greedy
    else:
        return {"k": 1}


if __name__ == '__main__':
    main()


