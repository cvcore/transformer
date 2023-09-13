from functools import lru_cache
import timeit
from typing import Dict, List, Optional
from collections import defaultdict
from unittest import result
import numpy as np
from tqdm import tqdm


class BytePairEncoder():

    START_TOKEN = '𝄆'
    END_TOKEN = '𝄇'
    PADDING_TOKEN = '♪'
    UNKNOWN_TOKEN = '☐'

    def __init__(
        self,
        language: str,
        max_vocab_size: int = 300,
        use_start_token: bool = False,
        use_end_token: bool = False,
        use_padding_token: bool = False,
        max_token_len: Optional[int] = None
    ) -> None:

        self.init_vocabulary(language)
        self._max_vocab_size = max_vocab_size
        self._token_pairs = defaultdict(int)

        self._use_start_token = use_start_token
        self._use_end_token = use_end_token
        self._use_padding_token = use_padding_token
        self._max_token_len = max_token_len

    def init_vocabulary(self, language: str):
        if language == 'english':
            vocabulary = [self.START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',  # fmt: skip
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        ':', '<', '=', '>', '?', '@',
                        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                        'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                        'Y', 'Z',
                        '[', '\\', ']', '^', '_', '`',
                        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                        'y', 'z',
                        '{', '|', '}', '~', self.PADDING_TOKEN, self.END_TOKEN, self.UNKNOWN_TOKEN]
        elif language == 'german':
            vocabulary = [self.START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        ':', '<', '=', '>', '?', '@',
                        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                        'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                        'Y', 'Z', 'Ä', 'Ö', 'Ü', 'ß',
                        '[', '\\', ']', '^', '_', '`',
                        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                        'y', 'z', 'ä', 'ö', 'ü',
                        '{', '|', '}', '~', self.PADDING_TOKEN, self.END_TOKEN, self.UNKNOWN_TOKEN]
        else:
            raise ValueError(f"Language {language} not supported. Must be either 'english' or 'german'.")

        self._vocabulary = vocabulary
        self._max_alphabet_len = max([len(token) for token in self._vocabulary])

    def add_to_vocabulary(self, token: str) -> None:
        if token in self._vocabulary:
            print("Alphabet already in vocabulary.")
            return
        self._vocabulary.append(token)
        # self.reset_cache()
        self._max_alphabet_len = max(self._max_alphabet_len, len(token))

    @lru_cache()
    def get_token_id(self, token: str) -> int:
        if token not in self._vocabulary:
            return self.get_token_id(self.UNKNOWN_TOKEN)
        return self._vocabulary.index(token)

    @lru_cache()
    def get_token_from_id(self, id: int) -> str:
        return self._vocabulary[id]

    def reset_cache(self):
        self.get_token_id.cache_clear()
        self.get_token_from_id.cache_clear()

    def decode_sentence(self, tokens: List[id]) -> str:
        return ''.join([self.get_token_from_id(id) for id in tokens])

    def encode_sentence(
        self,
        sentence: str
    ) -> List[int]:

        result = []

        max_token_len = self._max_token_len

        if self._use_start_token:
            sentence = self.START_TOKEN + sentence
        if self._use_end_token:
            sentence = sentence + self.END_TOKEN

        idx = 0
        while idx < len(sentence) and (max_token_len is None or len(result) < max_token_len):
            longest_alphabet = None

            for alphabet_len in range(min(self._max_alphabet_len, len(sentence) - idx), 0, -1):  # here we start with the longest possible token first.
                alphabet = sentence[idx:idx + alphabet_len]
                if alphabet in self._vocabulary:
                    longest_alphabet = alphabet
                    longest_alphabet_len = alphabet_len
                    break
                # else:
                #     break

            result.append(self.get_token_id(longest_alphabet))
            idx += longest_alphabet_len

        if self._use_padding_token:
            assert max_token_len is not None, "max_n_tokens must be specified when using padding."
            result.extend([self.get_token_id(self.PADDING_TOKEN)] * (max_token_len - len(result)))

        return result

    def encode_corpus(self, corpus: List[str]) -> np.ndarray:
        return np.array([self.encode_sentence(sentence) for sentence in corpus])

    def decode_corpus(self, tokens: np.ndarray) -> List[str]:
        return [self.decode_sentence(sentence) for sentence in tokens]

    def count_token_pairs(self, tokens: List[int]) -> None:
        for idx in range(len(tokens) - 1):
            if self._use_padding_token and (tokens[idx] == self.get_token_id(self.PADDING_TOKEN) or tokens[idx + 1] == self.get_token_id(self.PADDING_TOKEN)):
                break
            self._token_pairs[(tokens[idx], tokens[idx + 1])] += 1

    def learn_vocabulary_from_corpus(self, corpus: List[str]):
        progress = tqdm(total=self._max_vocab_size - len(self._vocabulary))

        while len(self._vocabulary) < self._max_vocab_size:
            self._token_pairs = defaultdict(int)

            for text in corpus:
                tokens = self.encode_sentence(text)
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


if __name__ == "__main__":

    test_text_corpus = ['This is a test sentence.', 'This is another test sentence.'] * 1000
    encoder = BytePairEncoder('english', max_vocab_size=300, use_start_token=True, use_end_token=True, use_padding_token=True, max_token_len=30)
    time = timeit.timeit(lambda: encoder.learn_vocabulary_from_corpus(test_text_corpus), number=1)
    print(f"Time: {time}s for learning vocabulary from {len(test_text_corpus)} sentences.")
    print(f"Final vocabulary size: {len(encoder._vocabulary)}")
    print(f"Final vocabulary: {encoder._vocabulary}")
