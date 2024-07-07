from .Models.Factory.llm_factory import LlmFactory
import os as __os
from .utils import get_config as __get_config

__config = __get_config()
__os.environ['HF_HOME'] = __config['hf_home']
__os.environ['SENTENCE_TRANSFORMERS_HOME'] = __config['st_home']

__version__ = '0.1.13'