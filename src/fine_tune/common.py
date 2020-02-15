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
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import GPT2Tokenizer, OpenAIGPTTokenizer, GPT2LMHeadModel, OpenAIGPTLMHeadModel


MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
}


def init_model(model_name: str,
               model_type: str,
               device):
    """
    Initialize a pre-trained LM
    :param model_name: from MODEL_CLASSES
    :param device: CUDA / CPU device
    :return: the model and tokenizer
    """
    logger.info(f'Initializing {model_name}')
    model_class, tokenizer_class = MODEL_CLASSES[model_type]
    tokenizer = tokenizer_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_ending(model: PreTrainedModel,
                    tokenizer: PreTrainedTokenizer,
                    prefix: str,
                    device: str,
                    num_samples: int = 1,
                    num_beams: int = 0,
                    p: float = 0.0,
                    k: int = 0,
                    temperature: float = 1.0,
                    length: int = 75):
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
    context_tokens = tokenizer.encode(prefix)
    num_return_sequences = num_beams if num_beams > 0 else num_samples
    num_beams = num_beams if num_beams > 0 else None
    k = k if k > 0 else None
    p = p if p > 0 else None

    # TODO: change to num_beams == 0 when huggingface support returning all beams
    # do_sample = True
    do_sample = num_beams is None
    max_length = length + len(context_tokens)

    input_ids = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(0)
    outputs = model.generate(
        input_ids=input_ids, max_length=max_length, do_sample=do_sample, temperature=temperature,
        num_return_sequences=num_return_sequences, num_beams=num_beams, top_p=p, top_k=k,
        eos_token_ids=stop_token)

    if num_return_sequences > 1:
        outputs = outputs[0]

    outputs = outputs[:, len(context_tokens):]
    texts = set([re.sub(" +", " ",
        tokenizer.decode(outputs[i], clean_up_tokenization_spaces=True).replace(
        "<eos>", "").replace("<unk>", "").replace("!", "").strip())
                 for i in range(num_return_sequences)])
    texts = [text for text in texts if len(text) > 2]
    return texts


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
    df = pd.read_csv(in_file, index_col=0)
    df.iloc[:, :9] = df.iloc[:, :9].apply(lambda col: col.apply(json.loads))
    examples = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        for cat in categories:
            if len(row[cat]) > 0:
                examples[row.name.lower()][cat] = f"{row.name.lower().replace('___', '<blank>')} <{cat}>"

    examples = {event: {cat: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prefix))
                        for cat, prefix in curr_categories.items()}
                for event, curr_categories in examples.items()}

    return examples


def load_atomic_data_for_eval(in_file, categories, tokenizer, max_input_length, max_output_length):
    """
    Loads an ATOMIC dataset file and
    :param in_file: CSV ATOMIC file
    :param categories: ATOMIC category list
    :param tokenizer: LM tokenizer
    :return: dictionary of category to dictionary of event string to list of prefixes for generation and events
    """
    df = pd.read_csv(in_file, index_col=0)
    df.iloc[:, :9] = df.iloc[:, :9].apply(lambda col: col.apply(json.loads))
    examples = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        for cat in categories:
            if len(row[cat]) > 0:
                examples[cat][row.name.lower()] = (f"{row.name.lower().replace('___', '<blank>')} <{cat}>", row[cat])

    examples = {cat: {event: (list(map(str.lower, e2s)),
                           tokenizer.convert_tokens_to_ids(tokenizer.tokenize(e1)),
                           [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{e2.lower()} <eos>")) for e2 in e2s])
                        for event, (e1, e2s) in curr_events.items()}
                for cat, curr_events in examples.items()}

    # Pad
    max_length = [max_input_length + 1, max_output_length + 1]
    examples = {cat: {event: (e2s,
                              tok_e1[:max_length[0]] + [0] * max(0, max_length[0] - len(tok_e1)),
                              [tok_e2[:max_length[1]] + [0] * max(0, max_length[1] - len(tok_e2)) for tok_e2 in tok_e2s])
                      for event, (e2s, tok_e1, tok_e2s) in curr_events.items()}
                for cat, curr_events in examples.items()}

    return examples


def load_atomic_data_for_training(in_file, categories, tokenizer, max_input_length, max_output_length):
    """
    Loads an ATOMIC dataset file and
    :param in_file: CSV ATOMIC file
    :param categories: ATOMIC category list
    :param tokenizer: LM tokenizer
    :return: a list of tuples
    """
    df = pd.read_csv(in_file, index_col=0)
    df.iloc[:, :9] = df.iloc[:, :9].apply(lambda col: col.apply(json.loads))

    examples = [(f"{row.name.lower().replace('___', '<blank>')} <{cat}>", f"{event.lower()} <eos>")
                for _, row in df.iterrows() for cat in categories
                for event in row[cat] if len(row[cat]) > 0]
    process = lambda s: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))
    examples = [tuple(map(process, ex)) for ex in examples]

    # Pad
    max_length = [max_input_length, max_output_length + 1]
    examples = [tuple([ex[i][:max_length[i]] + [0] * max(0, max_length[i] - len(ex[i]))
                       for i in range(2)])
                for ex in examples]
    examples = [tokenizer.build_inputs_with_special_tokens(*ex) for ex in examples]
    return examples
