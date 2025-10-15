import pickle
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


from tests.adapters import run_train_bpe

DATA_DIR = r""
INPUT_PATH = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train.txt")

# Tokenizer save path
TOKENIZER_DIR = r""
VOCAB_PATH = os.path.join(TOKENIZER_DIR,"tinystories_bpe_vocab.pkl")
MERGES_PATH = os.path.join(TOKENIZER_DIR,"tinystories_bpe_merges.pkl")

# parameters
vocab_size = 10_000
special_tokens = ["<|endoftext|>"]

vocab, merges = run_train_bpe(
    INPUT_PATH,
    vocab_size,
    special_tokens
)

os.makedirs(TOKENIZER_DIR,exist_ok=True)
with open(VOCAB_PATH,"wb") as f:
    pickle.dump(vocab, f)

with open(MERGES_PATH,"wb") as f:
    pickle.dump(merges, f)

