from typing import List, TYPE_CHECKING
import re
from .base_splitter import BaseTextSplitter
if TYPE_CHECKING:
    from ..Tokenizer.base_tokenizer import BaseTokenizer

def split_sentences(text: str) -> List[str]:
    # Adapted from https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences
    if text.strip() == '':
        return [text]

    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov|edu|me)"
    digits = "([0-9])"
    multiple_dots = r'\.{2,}'

    prefix_pattern = re.compile(r'^([\s\n]*)')
    def extract_prefix(text):
        match = prefix_pattern.match(text)
        if match:
            prefix = match.group(1)
            # The rest of the string is the original text minus the prefix
            rest = text[len(prefix):]
        else:
            prefix = ''
            rest = text
        return (prefix, rest)

    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text: text = text.replace(".”", "”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"", "\"!")
    if "?" in text: text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    prefix_and_sentences = [extract_prefix(t) for t in sentences]
    pref, new_sentences = list(zip(*prefix_and_sentences))
    pref = list(pref)[1:] + ['']
    sentences = sentences[:1] + list(new_sentences)[1:]
    sentences = [s + p for s, p in zip(sentences, pref)]
    return sentences

class SentenceTextSplitter(BaseTextSplitter):
    """Text splitter that split texts into sentences.
    """

    def __init__(self, tokenizer: "BaseTokenizer",
                 chunk_size: int = 400, chunk_overlap: int = 40) -> None:
        """Initialize the TextSplitter.

        Args:
            tokenizer (BaseTokenizer): An instance of the BaseTokenizer class.
            chunk_size (int, optional): The maximum number of tokens per text chunk. Defaults to 400.
            chunk_overlap (int, optional): The number of tokens that overlaps for each subsequent chunk. Defaults to 40.
        """
        self._tokenizer = tokenizer
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap



    def split_text(self, text: str) -> List[str]:
        """
        Splits the given text into a list of strings.

        Args:
            text (str): The text to split.

        Returns:
            List[str]: A list of strings resulting from the split.
        """
        sentences = split_sentences(text=text)
        token_ids = self._tokenizer.batch_tokenize(sentences, add_special_tokens=False)
        num_tokens = [len(tids) for tids in token_ids] # Count tokens by sentences just for a rough estimation
        chunks = []
        current_chunk = []
        current_count = 0
        last_count = 0
        for i, sent in enumerate(sentences):
            sent_ct = num_tokens[i]
            if (sent_ct + current_count) <= self._chunk_size:
                current_count += sent_ct
                current_chunk.append(sent)
                last_count = sent_ct
            else:
                if len(current_chunk) != 0:
                    chunks.append(''.join(current_chunk))
                current_chunk = [sentences[i - 1]] if ((last_count <= self._chunk_overlap) and (i != 0)) else []
                current_count = num_tokens[i - 1] if ((last_count <= self._chunk_overlap) and (i != 0)) else 0
                current_count += sent_ct
                current_chunk.append(sent)
                last_count = sent_ct
        if len(current_chunk) != 0:
            chunks.append(''.join(current_chunk))
        return chunks