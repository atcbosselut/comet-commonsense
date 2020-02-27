import re
import torch
import logging

from transformers.modeling_utils import PreTrainedModel
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers.tokenization_utils import PreTrainedTokenizer

from src.atomic import get_atomic_categories

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

logger = logging.getLogger(__name__)


class PretrainedCometModel(object):
    """
    Pretrained COMET model, used for generating predictions.
    """
    def __init__(self, model_name_or_path, device="cpu", do_lower_case=True):
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() and device != "cpu" else "cpu")
        logger.debug(f"Initializing {device}")

        self.tokenizer, self.model = init_model(model_name_or_path, device=device, do_lower_case=do_lower_case)
        self.categories = set(get_atomic_categories())

    def predict(self,
                input_event: str,
                category: str,
                num_samples: int = 1,
                num_beams: int = 0,
                p: float = 0.0,
                k: int = 0,
                temperature: float = 1.0,
                length: int = 75,
                return_tokenized: bool = False):

        if category not in self.categories:
            raise ValueError(f"Category {category} not supported. Select one of {self.categories}")

        prefix = f"{input_event} <{category}>"
        return generate_ending(self.model, self.tokenizer, prefix, device=self.device,
                               num_samples=num_samples, num_beams=num_beams, p=p, k=k, temperature=temperature,
                               length=length, return_tokenized=return_tokenized)


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

