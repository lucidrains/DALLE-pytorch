from tokenizers import Tokenizer
import torch

class HGTokenizer(object):
    def __init__(self, bpe_path = None):
      bpe_full_path = return os.path.join(os.path.dirname(os.path.abspath(__file__)), bpe_path)
      self.tokenizer = Tokenizer.from_file(bpe_full_path)
      self.vocab_size = tokenizer.get_vocab_size()
    def encode(self, text):
      return self.tokenizer.encode(text)  

def tokenize(texts, context_length = 256, add_start = False, add_end = False, truncate_text = False):
    if isinstance(texts, str):
      texts = [texts]
      sot_tokens = tokenizer.encode("<|startoftext|>").ids if add_start else []
      eot_tokens = tokenizer.encode("<|endoftext|>").ids if add_end else []
      all_tokens = [sot_tokens + tokenizer.encode(text).ids + eot_tokens for text in texts]
      result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
      for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
          if truncate_text:
            tokens = tokens[:context_length]
          else:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
          result[i, :len(tokens)] = torch.tensor(tokens)
      return result
