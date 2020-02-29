import re
import os
import torch

import torch.nn.functional as F

from transformers import AutoModelWithLMHead, AutoTokenizer

from comet2.atomic import get_atomic_categories

BASE_DIR = os.path.expanduser("~/.comet-data/")
MODEL_DIR = os.path.join(BASE_DIR, "models")


class PretrainedCometModel(object):
    """
    Pretrained COMET model, used for generating predictions.
    """
    def __init__(self,
                 model_name_or_path=os.path.join(MODEL_DIR, "atomic_pretrained_model_openai-gpt"),
                 device="cpu",
                 do_lower_case=True):
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.tokenizer, self.model = init_model(model_name_or_path, device=self.device, do_lower_case=do_lower_case)
        self.categories = set(get_atomic_categories())
        self.eos_token_id = self.tokenizer.encode("<eos>")[0]

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

        prefix = f"{input_event} <{category.lower()}>"

        with torch.no_grad():
            if num_beams > 0:
                return self.beam_search(
                    prefix, num_beams=num_beams,
                    max_length=length, return_tokenized=return_tokenized)

            else:
                return self.generate_by_sampling(
                    prefix, num_samples=num_samples, p=p, k=k, temperature=temperature,
                    max_length=length, return_tokenized=return_tokenized)

    def beam_search(self,
                    prefix: str,
                    max_length: int = 50,
                    num_beams=1,
                    return_tokenized: bool = False):
        """
        An implementation of beam search based on the original COMET code:
        https://github.com/atcbosselut/comet-commonsense/blob/85b1e0550193eb1ff110f05f86362daf58dd7ddf/
        scripts/generate/generate_atomic_beam_search.py

        Because in HuggingFace's implementation they only return the top 1 sequence
        in greedy beam search.
        """
        context_tokens = self.tokenizer.encode(prefix)
        input_ids = torch.tensor(context_tokens, device=self.device).unsqueeze(0)

        kill_mask = torch.ones(num_beams, num_beams).to(self.device) * 9000
        kill_mask[:, 0] = 0
        beam_seqs = input_ids.repeat(num_beams, 1)

        # First token
        lm_probs = F.log_softmax(self.model(input_ids)[0], dim=-1)[:, -1, :].squeeze()
        beam_lls, beam_toks = lm_probs.topk(num_beams)
        beam_losses = [beam_lls]

        ended = (beam_toks == self.eos_token_id).float()
        counts = (2 - ended)
        beam_seqs = torch.cat((beam_seqs, beam_toks.unsqueeze(-1)), 1)

        for _ in range(max_length - 1):

            # Next token for each beam
            lm_probs = F.log_softmax(self.model(beam_seqs)[0], dim=-1)[:, -1, :].squeeze()
            hyp_beam_lls, hyp_beam_toks = lm_probs.topk(num_beams)

            # Compute masks and expand beam
            expanded_ended = ended.unsqueeze(1).repeat(1, num_beams)
            hypothesis_mask = expanded_ended * kill_mask + (1 - expanded_ended)
            current_beam_lls = beam_lls.unsqueeze(1).repeat(1, num_beams).view(num_beams ** 2)

            # Compute losses of hypotheses, masking those that have ended
            hyp_beam_lls = (hyp_beam_lls.view(num_beams ** 2) * hypothesis_mask.view(-1)) + current_beam_lls

            # Get normalizer for sequences
            temp_counts = counts.unsqueeze(1).repeat(1, num_beams).view(num_beams ** 2)

            # Select best beams with lowest aggregate loss
            beam_lls, top_beam_idxs = (hyp_beam_lls / temp_counts).topk(num_beams)

            # Update placements in beam based on selecetion
            beam_losses = [i.index_select(0, top_beam_idxs // num_beams) for i in beam_losses]
            ended = ended.index_select(0, top_beam_idxs // num_beams)
            counts = temp_counts.index_select(0, top_beam_idxs)

            # Save beam losses
            beam_losses.append(beam_lls * counts)

            # Update beam tokens
            ended_mask = (1 - ended).long()
            end_replacement = (self.eos_token_id * ended).long()
            next_toks = hyp_beam_toks.view(-1)[top_beam_idxs]
            beam_toks = next_toks * ended_mask + end_replacement

            # Update ended and counts
            ended = ended + (beam_toks == self.eos_token_id).float() * (1 - ended)
            counts = counts + (1 - ended)

            # Update beam sequences
            beam_seqs = beam_seqs.t().repeat(num_beams, 1).t().contiguous().view(num_beams ** 2, -1)[
                top_beam_idxs]
            beam_seqs = torch.cat((beam_seqs, beam_toks.unsqueeze(-1)), 1)

            if (beam_toks == self.eos_token_id).sum().item() == num_beams:
                break

        beam_seqs = beam_seqs[:, len(context_tokens):]

        if return_tokenized:
            texts = [self.tokenizer.convert_ids_to_tokens(t) for t in beam_seqs]
            texts = [t[:t.index("<eos>")] if "<eos>" in t else t for t in texts]
            texts = [[w for w in t if w != "<unk>"] for t in texts]
            texts = [[w.replace("</w>", "") for w in t] for t in texts]
            beam_seqs = [tuple(t) for t in texts if len(t) > 0]
        else:
            beam_seqs = [
                re.sub(" +", " ", self.tokenizer.decode(
                    seq, clean_up_tokenization_spaces=True).replace(
                    "<eos>", "").replace("<unk>", "").replace("!", "").replace("<w/>", "").strip())
                for seq in beam_seqs]

        return beam_seqs

    def generate_by_sampling(self,
                             prefix: str,
                             num_samples: int = 1,
                             p: float = 0.0,
                             k: int = 0,
                             temperature: float = 1.0,
                             max_length: int = 50,
                             return_tokenized: bool = False):
        """
        Generate endings for the beginning of the text by sampling
        :param prefix: text on which the generation is conditioned
        :param p: p for nucleus sampling
        :param temperature: default = 1
        :param max_length: the maximum length to sample
        :return: the texts
        """
        k = k if k > 0 else None
        p = p if p > 0 else None

        context_tokens = self.tokenizer.encode(prefix)
        input_ids = torch.tensor(context_tokens, device=self.device).unsqueeze(0)
        max_length = max_length + len(context_tokens)
        outputs = self.model.generate(
            input_ids=input_ids, max_length=max_length, do_sample=True, temperature=temperature,
            num_return_sequences=num_samples, top_p=p, top_k=k, eos_token_ids=self.eos_token_id)

        outputs = outputs[:, len(context_tokens):]

        if return_tokenized:
            texts = [self.tokenizer.convert_ids_to_tokens(t) for t in outputs]
            texts = [t[:t.index("<eos>")] if "<eos>" in t else t for t in texts]
            texts = [[w for w in t if w != "<unk>"] for t in texts]
            texts = [[w.replace("</w>", "") for w in t] for t in texts]
            outputs = {tuple(t) for t in texts if len(t) > 0}
        else:
            outputs = set([re.sub(" +", " ",
                self.tokenizer.decode(outputs[i], clean_up_tokenization_spaces=True).replace(
                "<eos>", "").replace("<unk>", "").replace("!", "").replace("<w/>", "").strip())
                         for i in range(num_samples)])

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
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model

