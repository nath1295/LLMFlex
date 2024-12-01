# adapted from mlx-lm
import json
from functools import partial
from typing import List, Optional, Union
import os
from transformers import PreTrainedTokenizer
from huggingface_hub import hf_hub_download

REPLACEMENT_CHAR = "\ufffd"
SPECIAL_SPACE = "\u2581"

def _remove_space(x):
    if x and x[0] == " ":
        return x[1:]
    return x

class NaiveDetokenizer:

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self._tokenizer = tokenizer
        self.reset()

    def reset(self, num_seqs: Optional[int] = None) -> None:
        self.num_seqs = num_seqs
        self.texts = [] if self.num_seqs is None else [''] * self.num_seqs
        self.tokens = [] if self.num_seqs is None else [[]] * self.num_seqs
        self.current_tokens = [] if self.num_seqs is None else [[]] * self.num_seqs
        self.final_segments = None

    def add_tokens(self, token_ids: List[List[int]]) -> None:
        if self.num_seqs is None:
            self.reset(num_seqs=len(token_ids))
        elif len(token_ids) != self.num_seqs:
            raise Exception('Number of tokens does not match with number of existing sequences.')
        self.current_tokens = [tids + nids for tids, nids in zip(self.current_tokens, token_ids)]

    def finalize(self) -> None:
        new_texts = self._tokenizer.batch_decode(self.current_tokens)
        new_texts = [nt.rstrip(REPLACEMENT_CHAR) for nt in new_texts]
        self.tokens = [t + ct for t, ct in zip(self.tokens, self.current_tokens)]
        self.texts = [t + nt for t, nt in zip(self.texts, new_texts)]
        self.current_tokens = [[]] * self.num_seqs
        self.final_segments = new_texts        

    @property
    def last_segments(self) -> List[str]:
        if self.final_segments is not None:
            return self.final_segments
        new_texts = self._tokenizer.batch_decode(self.current_tokens)
        with_repl = [newt.endswith(REPLACEMENT_CHAR) for newt in new_texts]
        bundle = [(ot if wr else ot + nt, nt if wr else [], os if wr else os + ns, '' if wr else ns) for ot, nt, os, ns, wr in zip(self.tokens, self.current_tokens, self.texts, new_texts, with_repl)]
        tokens, current_tokens, texts, new_segments = list(zip(*bundle))
        self.tokens = list(tokens)
        self.current_tokens = list(current_tokens)
        self.texts = list(texts)
        new_segments = list(new_segments)
        return new_segments

class SPMDetokenizer:
    """A streaming detokenizer for SPM models.

    It adds tokens to the text if the next token starts with the special SPM
    underscore which results in linear complexity.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, trim_space=True):
        self.trim_space = trim_space
        self.eos_token = tokenizer.eos_token

        # Extract the tokens in a list from id to text
        self.tokenmap = [""] * (max(tokenizer.vocab.values()) + 1)
        for value, tokenid in tokenizer.vocab.items():
            self.tokenmap[tokenid] = value

        self.hexcode_tokens = [i for i, t in enumerate(self.tokenmap) if t.startswith('<0x')]

        self.reset()

    def reset(self, num_seqs: Optional[int] = None) -> None:
        self.num_seqs = num_seqs
        self.range = [] if self.num_seqs is None else list(range(self.num_seqs))
        self.texts = [] if self.num_seqs is None else [''] * self.num_seqs
        self.tokens = [] if self.num_seqs is None else [[]] * self.num_seqs
        self.hexcodes = [] if self.num_seqs is None else [[]] * self.num_seqs
        self.segments = [] if self.num_seqs is None else [''] * self.num_seqs

    def _get_text_token(self, token_id, raw_token, token_condition, index) -> str:
        self.tokens[index].append(token_id[0])
        is_space, is_hex = token_condition
        output = ''
        if is_hex:
            self.hexcodes[index].append(int(raw_token[3:5], 16))
        elif is_space:
            if self.texts[index] or (not self.trim_space):
                output = ('' if len(self.hexcodes[index]) == 0 else bytes(self.hexcodes[index]).decode()) + raw_token.replace(SPECIAL_SPACE, ' ')
            else:
                output = ('' if len(self.hexcodes[index]) == 0 else bytes(self.hexcodes[index]).decode()) + _remove_space(raw_token.replace(SPECIAL_SPACE, ' '))
            self.hexcodes[index] = []
        else:
            output = ('' if len(self.hexcodes[index]) == 0 else bytes(self.hexcodes[index]).decode()) + raw_token
            self.hexcodes[index] = []
        self.texts[index] += output
        return output

    def add_tokens(self, token_ids: List[List[int]]) -> None:
        if self.num_seqs is None:
            self.reset(num_seqs=len(token_ids))
        elif len(token_ids) != self.num_seqs:
            raise Exception('Number of tokens does not match with number of existing sequences.')
        raw_tokens = [self.tokenmap[token[0]] for token in token_ids]
        token_conditions = [((rt[0] == SPECIAL_SPACE), (tid[0] in self.hexcode_tokens)) for rt, tid in zip(raw_tokens, token_ids)]
        self.segments = [self._get_text_token(tid, rt, tc, i) for tid, rt, tc, i in zip(token_ids, raw_tokens, token_conditions, self.range)]

    def finalize(self):
        hex_str = [('' if len(self.hexcodes[index]) == 0 else bytes(self.hexcodes[index]).decode()) for index in self.range]
        self.segments = [s + hs for s, hs in zip(self.segments, hex_str)]

    @property
    def last_segments(self) -> List[str]:
        """Return the last segment of readable text since last time this property was accessed."""
        segments = self.segments
        self.segments = [''] * self.num_seqs
        return segments

def _match(a, b):
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        return len(a) == len(b) and all(k in b and _match(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_match(ai, bi) for ai, bi in zip(a, b))
    return a == b

def _is_spm_decoder(decoder):
    _target_description = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0},
        ],
    }
    return _match(_target_description, decoder)

def _is_spm_decoder_no_space(decoder):
    _target_description = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
        ],
    }
    return _match(_target_description, decoder)

def get_detokenizer(pretrained_model_name_or_path: str, tokenizer: PreTrainedTokenizer) -> Union[NaiveDetokenizer, SPMDetokenizer]:
    """Get the detokenizer.

    Note, to use a fast streaming tokenizer, pass a local file path rather than
    a Hugging Face repo ID.
    """
    from ..utils import get_config
    detokenizer_class = NaiveDetokenizer

    tokenizer_file = os.path.join("tokenizer.json")
    if not os.path.exists(tokenizer_file):
        tokenizer_file = hf_hub_download(repo_id=pretrained_model_name_or_path, filename='tokenizer.json', cache_dir=get_config('hf_home'))
    with open(tokenizer_file, "r") as fid:
        tokenizer_content = json.load(fid)
    if "decoder" in tokenizer_content:
        if _is_spm_decoder(tokenizer_content["decoder"]):
            detokenizer_class = SPMDetokenizer
        elif _is_spm_decoder_no_space(tokenizer_content["decoder"]):
            detokenizer_class = partial(SPMDetokenizer, trim_space=False)
    return detokenizer_class(tokenizer=tokenizer)
    
    
