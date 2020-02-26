import re
import json
import torch
import logging
import pandas as pd
import utils.utils as utils
import src.data.config as cfg

from collections import defaultdict


logging.basicConfig(
    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)

logger = logging.getLogger(__name__)


from transformers.modeling_utils import PreTrainedModel
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers.tokenization_utils import PreTrainedTokenizer


def init_model(model_name: str,
               device,
               do_lower_case: bool = False):
    """
    Initialize a pre-trained LM
    :param model_name: from MODEL_CLASSES
    :param device: CUDA / CPU device
    :return: the model and tokenizer
    """
    logger.info(f'Initializing {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def generate_ending(model: PreTrainedModel,
                    tokenizer: PreTrainedTokenizer,
                    prefix: str,
                    device: torch.device,
                    num_samples: int = 1,
                    num_beams: int = 0,
                    p: float = 0.0,
                    k: int = 0,
                    temperature: float = 1.0,
                    length: int = 75,
                    return_tokenized: bool = False):
    """
    Generate an ending for the beginning of the text
    :param model: the pre-trained LM
    :param tokenizer: the pre-trained tokenizer
    :param prefix: text on which the generation is conditioned
    :param device: CUDA / CPU device
    :param p: p for nucleus sampling
    :param temperature: default = 1
    :param length: the maximum length to sample
    :return: the text
    """
    stop_token = tokenizer.encode("<eos>")[0]
    num_return_sequences = num_beams if num_beams > 0 else num_samples
    num_beams = num_beams if num_beams > 0 else None
    k = k if k > 0 else None
    p = p if p > 0 else None

    context_tokens = tokenizer.encode(prefix)

    # TODO: change to num_beams == 0 when huggingface support returning all beams
    do_sample = True
    # do_sample = num_beams is None
    max_length = length + len(context_tokens)

    input_ids = torch.tensor(context_tokens, device=device).unsqueeze(0)
    outputs = model.generate(
        input_ids=input_ids, max_length=max_length, do_sample=do_sample, temperature=temperature,
        num_return_sequences=num_return_sequences, num_beams=num_beams, top_p=p, top_k=k,
        eos_token_ids=stop_token)

    if num_return_sequences > 1:
        outputs = outputs[0]

    outputs = outputs[:, len(context_tokens):]

    if return_tokenized:
        new_outputs = set()
        for text in outputs:
            text = tokenizer.convert_ids_to_tokens(text)

            if "<eos>" in text:
                text = text[:text.index("<eos>")]

            if "<unk>" in text:
                text.remove("<unk>")

            if len(text) > 0:
                new_outputs.add(tuple(text))

        outputs = new_outputs
    else:
        outputs = set([re.sub(" +", " ",
            tokenizer.decode(outputs[i], clean_up_tokenization_spaces=True).replace(
            "<eos>", "").replace("<unk>", "").replace("!", "").strip())
                     for i in range(num_return_sequences)])

        outputs = [text for text in outputs if len(text) > 0]

    return list(outputs)


def get_atomic_categories(eval_mode=True):
    """
    Return the names of ATOMIC categories
    """
    utils.generate_config_files("atomic", "0", eval_mode=eval_mode)
    config_file = "config/atomic/config_0.json"
    config = cfg.read_config(cfg.load_config(config_file))
    opt, _ = cfg.get_parameters(config)
    return opt.data.categories


def load_atomic_data_for_generation(in_file, categories, tokenizer):
    """
    Loads an ATOMIC dataset file and
    :param in_file: CSV ATOMIC file
    :param categories: ATOMIC category list
    :param tokenizer: LM tokenizer
    :return: dictionary of event string to dictionary of category to list of prefixes for generation
    """
    examples = load_atomic_data(in_file, categories)
    examples = {e1: {cat: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{e1} <{cat}>"))
                     for cat in e1_relations.keys()}
                for e1, e1_relations in examples.items()}

    return examples


def load_atomic_data_for_eval(in_file, categories, tokenizer):
    """
    Loads an ATOMIC dataset file and
    :param in_file: CSV ATOMIC file
    :param categories: ATOMIC category list
    :param tokenizer: LM tokenizer
    :return: dictionary of category to dictionary of event string to list of prefixes for generation and events
    """
    examples = load_atomic_data(in_file, categories)
    examples_by_category = defaultdict(lambda: defaultdict(list))

    for e1, curr_relations in examples.items():
        for cat, e2s in curr_relations.items():
            examples_by_category[cat][e1.replace("<blank>", "___")] = (
                tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{e1} <{cat}>")), [f"{e2} <eos>" for e2 in e2s])

    return examples_by_category


def load_atomic_data_for_training(in_file, categories, tokenizer, max_input_length, max_output_length):
    """
    Loads an ATOMIC dataset file and
    :param in_file: CSV ATOMIC file
    :param categories: ATOMIC category list
    :param tokenizer: LM tokenizer
    :return: a list of tuples
    """
    examples = load_atomic_data(in_file, categories)
    examples = [(f"{e1} <{cat}>", f"{e2} <eos>")
                for e1, e1_relations in examples.items()
                for cat, e2s in e1_relations.items()
                for e2 in e2s]

    process = lambda s: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))
    examples = [tuple(map(process, ex)) for ex in examples]

    # Pad
    max_input_length = min(max_input_length, max([len(ex[0]) for ex in examples]))
    max_output_length = min(max_output_length, max([len(ex[1]) for ex in examples]))
    max_length = max_input_length + max_output_length + 1
    input_lengths = [len(ex[0]) for ex in examples]

    examples = [ex[0] + ex[1] for ex in examples]
    examples = [ex[:max_length] + [0] * max(0, max_length - len(ex)) for ex in examples]

    examples = {"examples": examples, "input_lengths": input_lengths}
    return examples


def load_atomic_data(in_file, categories):
    """
    Load ATOMIC data from the CSV file
    :param in_file: CSV file
    :param categories: list of ATOMIC categories
    :return: list of tuples: (e1 and catgory, e2)
    """
    df = pd.read_csv(in_file, index_col=0)
    df.iloc[:, :len(categories)] = df.iloc[:, :len(categories)].apply(lambda col: col.apply(json.loads))
    df = df.groupby("event").agg({cat: "sum" for cat in categories})

    examples = {row.name.lower().replace('___', '<blank>'): {
        cat: [e2.lower() for e2 in set(row[cat])] for cat in categories if len(row[cat]) > 0}
        for _, row in df.iterrows()}

    return examples