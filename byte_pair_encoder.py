from functools import lru_cache
from math import ceil
from pathlib import Path
import pickle
import timeit
import token
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
from unittest import result
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from trie import Trie
NoneType = type(None)


def merge_dictionary(*dictionaries):
    """Merge the dictionaries given as arguments and return a single dictionary without duplicated token.
    The ids in the first given dictionary will be preserved, and the others not."""

    final_dict = []
    for dict in dictionaries:
        for token in dict:
            if token not in final_dict:
                final_dict.append(token)

    return final_dict


class BytePairEncoder():

    # We use some uncommon symbols to represent the special tokens.
    START_TOKEN = '𝄆'
    END_TOKEN = '𝄇'
    PADDING_TOKEN = '♪'
    UNKNOWN_TOKEN = '☐'
    RESERVED = '♫'

    # Signal tokens are used to mark the beginning and end of a sentence and won't appear inside a normal sentence.
    # And we don't allow them to form a byte-pair.
    SIGNAL_TOKENS = [START_TOKEN, END_TOKEN, PADDING_TOKEN]

    def __init__(
        self,
        language: str,
        max_vocab_size: int = 300,
        use_start_token: bool = False,
        use_end_token: bool = False,
        use_padding_token: bool = False,
        max_token_len: Optional[int] = None,
        load_vocabulary_from: str = None,
    ) -> None:

        self._language = language
        if load_vocabulary_from is None:
            self.init_vocabulary(language)
        else:
            self.load_vocabulary(load_vocabulary_from)
        self._max_vocab_size = max_vocab_size
        self._token_pairs = defaultdict(int)

        self._use_start_token = use_start_token
        self._use_end_token = use_end_token
        self._use_padding_token = use_padding_token
        self._max_token_len = max_token_len

    def init_vocabulary(self, language: str):
        EN_VOCABULARY = [
            self.RESERVED, self.START_TOKEN, self.PADDING_TOKEN, self.END_TOKEN, self.UNKNOWN_TOKEN,
            ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            ':', '<', '=', '>', '?', '@',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
            'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
            'Y', 'Z',
            '[', '\\', ']', '^', '_', '`',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
            'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
            'y', 'z',
            '{', '|', '}', '~'
        ]

        DE_VOCABULARY = [
            self.RESERVED, self.START_TOKEN, self.PADDING_TOKEN, self.END_TOKEN, self.UNKNOWN_TOKEN,
            ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            ':', '<', '=', '>', '?', '@',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
            'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
            'Y', 'Z', 'Ä', 'Ö', 'Ü', 'ß',
            '[', '\\', ']', '^', '_', '`',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
            'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
            'y', 'z', 'ä', 'ö', 'ü',
            '{', '|', '}', '~', '§',
        ]

        if language == 'english':
            vocabulary = EN_VOCABULARY
        elif language == 'german':
            vocabulary = DE_VOCABULARY
        elif language == 'universal':
            vocabulary = merge_dictionary(EN_VOCABULARY, DE_VOCABULARY)

        else:
            raise ValueError(f"Language {language} not supported. Must be either 'english' or 'german' or 'universal'.")

        # self._vocabulary = vocabulary
        self._trie = Trie()
        for token in vocabulary:
            self._trie.insert(token)

    def add_to_vocabulary(self, token: str) -> None:
        if token in self._trie:
            print("Alphabet already in vocabulary.")
            return
        self._trie.insert(token)
        # self.reset_cache()
        # self._max_alphabet_len = max(self._max_alphabet_len, len(token))

    @lru_cache()
    def get_token_id(self, token: str) -> int:
        if token is None or token not in self._trie:
            return self.get_token_id(self.UNKNOWN_TOKEN)
        return self._trie.get_id(token)

    @lru_cache()
    def get_token_from_id(self, id: Union[NoneType, int]) -> str:
        if id is None or id < 0 or id >= len(self._trie):
            return self.UNKNOWN_TOKEN
        return self._trie.get_token(id)

    def reset_cache(self):
        self.get_token_id.cache_clear()
        self.get_token_from_id.cache_clear()

    def decode_sentence(self, tokens: List[id]) -> str:
        return ''.join([self.get_token_from_id(id) for id in tokens])

    def _encode_sentence_impl(
        self,
        sentence: str,
        max_token_len: Optional[int] = None,
        use_start_token: bool = False,
        use_end_token: bool = False,
        use_padding_token: bool = False,
    ):
        result = []

        if use_start_token:
            sentence = self.START_TOKEN + sentence
        if use_end_token:
            sentence = sentence + self.END_TOKEN

        idx = 0
        while idx < len(sentence) and (max_token_len is None or len(result) < max_token_len):
            longest_token_id = self._trie.longest_match(sentence, idx)
            if longest_token_id is None:
                longest_token_id = self.get_token_id(self.UNKNOWN_TOKEN)
                longest_token_len = 1
            else:
                longest_token_len = len(self.get_token_from_id(longest_token_id))
            result.append(longest_token_id)
            idx += longest_token_len

        if use_padding_token:
            assert max_token_len is not None, "max_n_tokens must be specified when using padding."
            result.extend([self.get_token_id(self.PADDING_TOKEN)] * (max_token_len - len(result)))

        return result

    def encode_sentence(
        self,
        sentence: str
    ) -> List[int]:
        return self._encode_sentence_impl(
            sentence,
            self._max_token_len,
            self._use_start_token,
            self._use_end_token,
            self._use_padding_token
        )

    def encode_corpus(self, corpus: List[str]) -> np.ndarray:
        return np.array([self.encode_sentence(sentence) for sentence in corpus]).astype(int)

    def decode_corpus(self, tokens: np.ndarray) -> List[str]:
        return [self.decode_sentence(sentence) for sentence in tokens]

    def count_token_pairs(self, tokens: List[int]) -> None:
        for idx in range(len(tokens) - 1):
            self._token_pairs[(tokens[idx], tokens[idx + 1])] += 1

    def _count_pairs_parallel(self, corpus: List[str], start_idx, end_idx) -> Dict[Tuple, int]:
        token_pairs = defaultdict(int)

        for text in corpus[start_idx:end_idx]:
            tokens = self._encode_sentence_impl(text, use_start_token=False, use_end_token=False, use_padding_token=False, max_token_len=None)
            for idx in range(len(tokens) - 1):
                token_pairs[(tokens[idx], tokens[idx + 1])] += 1

        return token_pairs

    def learn_vocabulary_from_corpus(self, corpus: List[str], n_processes: int = 1):
        progress = tqdm(total=self._max_vocab_size - len(self._trie))

        while len(self._trie) < self._max_vocab_size:
            self._token_pairs = defaultdict(int)

            if n_processes > 1:
                chunk_size = int(ceil(len(corpus) / n_processes))
                chunk_indices = [(i * chunk_size, min((i + 1) * chunk_size, len(corpus))) for i in range(n_processes)]
                with Pool(n_processes) as pool:
                    results = pool.starmap(self._count_pairs_parallel, [(corpus, start_idx, end_idx) for (start_idx, end_idx) in chunk_indices])
                for result in results:
                    for key, value in result.items():
                        self._token_pairs[key] += value
            else:  # no parallelization
                for text in corpus:
                    # When learning the vocabulary, focus on the sentence content.
                    tokens = self._encode_sentence_impl(text, use_start_token=False, use_end_token=False, use_padding_token=False, max_token_len=None)
                    self.count_token_pairs(tokens)

            token_pairs_sorted = sorted(self._token_pairs.items(), key=lambda x: x[1], reverse=True)
            if len(token_pairs_sorted) == 0:
                # At this point, the whole corpus got compressed to a single token.
                print("Maximum compression reached. Stopping.")
                break
            most_freq_pair = token_pairs_sorted[0][0]
            new_token = self.get_token_from_id(most_freq_pair[0]) + self.get_token_from_id(most_freq_pair[1])

            self.add_to_vocabulary(new_token)

            progress.update(1)

    def save_vocabulary(self, path: str):
        vocabulary_data = {
            'language': self._language,
            'trie': self._trie,
            'max_vocab_size': self._max_vocab_size,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Saving vocabulary data to: {path}")
        with open(path, "wb") as pk_file:
            pickle.dump(vocabulary_data, pk_file)

    def load_vocabulary(self, path: str):
        print(f"[INFO] Loading vocabulary data from: {path}")
        with open(path, "rb") as pk_file:
            vocabulary_data = pickle.load(pk_file)
        for key, value in vocabulary_data.items():
            setattr(self, f"_{key}", value)


if __name__ == "__main__":

    test_text_corpus = ['This is a test sentence.', 'This is another test sentence.', "Das ist ein Löffel"] * 1000
    if Path('data/vocabulary.pkl').exists():
        encoder = BytePairEncoder('universal', max_vocab_size=300, use_start_token=True, use_end_token=True, use_padding_token=True, max_token_len=30, load_vocabulary_from='data/vocabulary.pkl')
    else:
        encoder = BytePairEncoder('universal', max_vocab_size=300, use_start_token=True, use_end_token=True, use_padding_token=True, max_token_len=30)
        time = timeit.timeit(lambda: encoder.learn_vocabulary_from_corpus(test_text_corpus, n_processes=8), number=1)
        encoder.save_vocabulary('data/vocabulary.pkl')
        print(f"Time: {time}s for learning vocabulary from {len(test_text_corpus)} sentences.")

    encoder_without_signals = BytePairEncoder('universal', max_vocab_size=300, use_start_token=False, use_end_token=False, use_padding_token=False, max_token_len=30, load_vocabulary_from='data/vocabulary.pkl')
    print(f"Final vocabulary size: {len(encoder._trie)}")
    print(f"Final vocabulary: {encoder._trie._tokens}")
    print(f"\"I am a Transformer.\" => {encoder.encode_sentence('I am a Transformer.')}")
    print(f"\"Deutschland ÄÖÜ\" => {encoder.encode_sentence('Deutschland ÄÖÜ')}")
    print(f"\"This is a lonnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnng string.......\" => {encoder.encode_sentence('This is a lonnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnng string.......')}")

    print(f"\"I am a Transformer.\" => {encoder_without_signals.encode_sentence('I am a Transformer.')}")
