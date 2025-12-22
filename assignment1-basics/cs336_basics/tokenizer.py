import regex as re
from typing import Iterable, Iterator

# pre-tokenize GPT-2, GPT-3 used.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.byte_to_token_id = {v:k for k,v in vocab.items()}
        self.merges = merges

        self.bpe_rank = dict(zip(merges,range(len(merges)))) # merge and corresponding ids; smaller id-->merge earlyer when training = merge earlyer when tokenizing

        # Special Tokens
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens_bytes = [token.encode("utf-8") for token in self.special_tokens]
        
        # Ensure special tokens are in vocab
        for special_byte in self.special_tokens_bytes:
            if special_byte not in self.byte_to_token_id:
                special_id = len(self.vocab)
                self.byte_to_token_id[special_byte] = special_id
                self.vocab[special_id] = special_byte

    def encode(
        self,
        text:str
    ) -> list[int]:
        """
        Encode the text into a sequence of token ids.
        
        Args:
            text: Input to encode.

        Returns:
            A sequnce of token ids.
        """
        tokens = []

        # special_tokens = self.special_tokens  !! sort by len, avoiding partial matches
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "|".join(map(re.escape,sorted_special_tokens))
        if pattern:
            parts = re.split(f"({pattern})",text)
        else:
            parts = [text]
        
        for part in parts:
            if part in self.special_tokens:
                tokens.append(self.byte_to_token_id[part.encode("utf-8")])  # string to UTF-8 encoded bytes
            else:
                # tokens.extend(self._tokenize(part))
                sub_parts = re.findall(PAT, part)
                for sub_part in sub_parts:
                    tokens.extend(self._tokenize(sub_part))  # tokenize every word
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        """
        Class method that loads a tokenizer from serialized files.

        Args:
            vocab_filepath: Path to a pickle file containing a dict[int, bytes] (or bytes-like).
            merges_filepath: Path to a pickle file containing a list[tuple[bytes, bytes]] (or str-like).
            special_tokens: Optional list[str] to be registered/appended to the vocabulary.

        Returns:
            An initialized Tokenizer instance.
        """
        import pickle

        # Load and normalize vocab: keys -> int, values -> bytes
        with open(vocab_filepath, "rb") as vf:
            raw_vocab = pickle.load(vf)

        norm_vocab: dict[int, bytes] = {}
        for k, v in raw_vocab.items():
            kid = int(k)
            if isinstance(v, str):
                v = v.encode("utf-8")
            norm_vocab[kid] = v

        # Load and normalize merges: ensure tuples of bytes
        with open(merges_filepath, "rb") as mf:
            raw_merges = pickle.load(mf)

        norm_merges: list[tuple[bytes, bytes]] = []
        for a, b in raw_merges:
            if isinstance(a, str):
                a = a.encode("utf-8")
            if isinstance(b, str):
                b = b.encode("utf-8")
            norm_merges.append((a, b))

        return cls(norm_vocab, norm_merges, special_tokens)

    def decode(
        self,
        ids: list[int]
    ) -> str:
        """
        Decode a sequence of integer token ids into a string.
        Args:
            ids: A list of integer token ids.
        
        Returns:
            decoded string.
        """
        bytes = b''.join(self.vocab[token_id] for token_id in ids)
        return bytes.decode("utf-8",errors="replace")


    def _tokenize(
           self,
           text:str 
    ) -> list[int]:
        """
        tokenize a string sequence without special tokens into token ids.

        Args:
        text: Input to tokenize.

        Return:
            A sequnce of token ids.
        """
        pre_tokens = []
        for m in re.finditer(PAT,text):
            word = m.group(0)
            pre_tokens.append(word)
        
        def word_2_byte(word: str) -> tuple[bytes, ...]:
            word_decoded = list(word.encode('UTF-8'))
            #split the bytes
            word_byte = [bytes([b]) for b in word_decoded]
            return tuple(word_byte)

        token_ids = []
        for token in pre_tokens:
            # convert token to bytes
            byte_tuple = word_2_byte(token)
            
            # BPE merges: bytes pair merges
            merged = self._merges(byte_tuple)

            # token IDs
            token_ids.extend(self.byte_to_token_id[b] for b in merged)
        
        return token_ids


    def _merges(
            self,
            byte_tuple: tuple[bytes]
    ) -> list[bytes]:
        """
        Apply BPE merges to byte_tuple

        Args:
            byte_tuple: tuple of single-byte token

        Returns:
            List of merged byte tokens after BPE merging.
        """
        word = list(byte_tuple)

        def get_pairs(word:list[bytes]): #byte pair
            pairs = set()
            prev_char = word[0]
            for char in word[1:]:
                pairs.add((prev_char,char))
                prev_char = char
            return pairs
        
        pairs = get_pairs(word)

        if not pairs:
            return word
        
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_rank.get(pair,float('inf'))) 
            # Get the merge priority, return the pair itself with the smallest bpe_ranks value, not the minimum value.
            if bigram not in self.bpe_rank:    # inf
                break

            first, second = bigram              # get tuple (b1,b2)

            new_word = []

            i = 0
            while i < len(word):
                try:
                    j = word.index(first,i)     # get i-th b1 index
                except ValueError:
                    new_word.extend(word[i:])   # can't find, add left to new_word
                    break
                else:
                    new_word.extend(word[i:j])  # i~j tokens
                    i = j
                
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_word = tuple(new_word)          # merged bytes
            word = new_word
            if len(word) == 1:
                break                           # stop and return
            else:
                pairs = get_pairs(word)
            
        return word