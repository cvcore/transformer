"""In order to speed-up the vocabulary building speed of the BPE algorithm, we use a prefix tree (aka. Trie) data structure to index the vocabulary.
In a Trie, each node contains one character of a token, and its children nodes contain the next character of the token.
The root node is an empty character. When we encounter a node with a valid `token_id` when traversing the Trie, we have found a valid token.
"""
from dataclasses import dataclass
from typing import Dict, Union
NoneType = type(None)


@dataclass
class TrieNode():
    children: Dict
    token_id: Union[NoneType, int]


class Trie:
    def __init__(self) -> None:
        self._root = TrieNode(children={}, token_id=None)
        self._tokens = []

    def insert(self, token: str) -> int:
        """Insert a token into the Trie. Return the token's id."""
        node = self._root

        for char in token:
            if char not in node.children:
                node.children[char] = TrieNode(children={}, token_id=None)
            node = node.children[char]
        if node.token_id is None:
            node.token_id = len(self._tokens)
            self._tokens.append(token)
        return node.token_id

    def get_id(self, token: str) -> Union[NoneType, int]:
        """Return the token's id if it is in the Trie, otherwise return None."""
        node = self._root

        for char in token:
            if char not in node.children:
                return None
            node = node.children[char]
        return node.token_id

    def get_token(self, token_id: int) -> str:
        """Return the token corresponding to the token_id."""
        assert token_id < len(self._tokens)
        return self._tokens[token_id]

    def longest_match(self, s: str, from_idx: int = 0) -> Union[int, NoneType]:
        """Return the longest token id that matches the string `s` starting from index `from_idx`."""
        node = self._root
        longest_match = None

        for i in range(from_idx, len(s)):
            char = s[i]
            if char not in node.children:
                break
            node = node.children[char]
            if node.token_id is not None:
                longest_match = node.token_id
        return longest_match

    def __len__(self) -> int:
        return len(self._tokens)

    def __contains__(self, token: str) -> bool:
        return self.get_id(token) is not None


if __name__ == "__main__":
    trie = Trie()
    print(trie.insert("a"))
    print(trie.insert("abc"))
    print(trie.insert("你"))
    print(trie.insert("你好吗"))
    print(trie._tokens)
    print(trie.get_id("abc"))
    print(trie.get_id("ab"))
    print(trie.get_id("你好"))
    print(trie.insert("你好"))
    print(trie.get_id("你好"))
    print(trie.longest_match("你好吗abc你好吗"))
    print(trie.longest_match("你好吗"))
