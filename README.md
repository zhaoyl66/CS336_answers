# Stanford CS336 Assignment Solutions
This project is my answers for Stanford CS336 Assignment.
## Assignment 1 
### Implementing the BPE training

#### Pre-tokenization
- Parallel pre-tokenization can efficiently process large input corpora through streaming

#### BPE Merge
- The key to BPE training is dynamically maintaining byte pairs, tokens, and their respective occurrence counts

#### test 
- implement the test adapter at [adapters.run_train_bpe], and then test using:
```bash
uv run pytest tests/test_train_bpe.py

### Implementing the tokenizer
- Do not skip pre-tokenization—applying BPE directly to full sentences corrupts encoding by merging bytes across word boundaries.
- Do not use special tokens in random order—always sort them by length (longest first) to prevent shorter tokens from incorrectly matching parts of longer ones.
- Do not ignore edge cases—always handle inputs like empty strings explicitly to ensure basic functionality passes tests.

#### test 
- implement the test adapter at [cs336_basics.tokenizer], and then test using:
```bash
uv run pytest tests/test_tokenizer.py

