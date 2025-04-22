import collections
from typing import Optional
import json
import os

def create_word_freq_dict(Text: list) -> tuple[dict, dict]:
    word_freq = collections.defaultdict(int) # Computes count of each individual words.
    for text in Text:
        new_words = text.split()
        for word in new_words:
            word = " ".join(word.strip()) + " </w>" # Creates space between characters of word. </w> indicates the end of word.
            word_freq[word] += 1

    """Creates a vocabulary of individual characters along with count"""
    vocab_count_dic = collections.defaultdict(int)

    for word, freq in word_freq.items():
        for letter in word.split():
                vocab_count_dic[letter] += freq

    return word_freq, vocab_count_dic
    

"""Function to compute the character pair along with counts"""
def compute_pair_freqs(word_freq: dict, splits: dict) -> dict:
    pair_freqs = collections.defaultdict(int)
    for word, freq in word_freq.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


"""Function to merge the best pairs"""
def merge_pair(a: str, b: str, word_freq: dict, splits: dict) -> dict:
    for word in word_freq:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[: i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits


class BPE:
    
    def create_tokenizer(self, 
                         Text: list, 
                         vocab_size: int = 30_000, 
                         max_iterations: int = 100, 
                         tokenizer_file_name: Optional[str] = "Tokenizer",
                         pad_token: Optional[str] = "<pad>", 
                         special_tokens: Optional[list] = []) -> None:
        r"""
        Function to create byte-pair encoding tokenizer in json.
        Parameters:
            Text (list): List of texts
            vocab_size (int): Maximum size of vocabulary (default: 30000)
            max_iterations (int): Maximum number of iterations to refine the vocabulary (default: 100)
            tokenizer_file_name (str): Name of the json to be saved (default: "Tokenizer")
            pad_token (str): Pad token (default: "\<pad\>")
            special_tokens (list): List of special tokens to be added to the vocabulary (default: [])

        """

        word_freq, vocab_count_dic = create_word_freq_dict(Text)
        splits = {word: word.split() for word in word_freq.keys()} # 1
        pair_freqs = compute_pair_freqs(word_freq, splits) 
        merges = {}
        i = 0
        while i < max_iterations: # For max iterations
            pair_freqs = compute_pair_freqs(word_freq, splits) # 2
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key = pair_freqs.get) # 3
            key = best_pair[0] + ":" + best_pair[1]
            merges[key] = best_pair[0] + best_pair[1] # 4
            # 5
            max_freq = pair_freqs[best_pair]
            vocab_count_dic[best_pair[0] + best_pair[1]] = max_freq
            vocab_count_dic[best_pair[0]] -= max_freq
            vocab_count_dic[best_pair[1]] -= max_freq
            if vocab_count_dic[best_pair[0]] == 0: 
                vocab_count_dic.pop(best_pair[0])
            if vocab_count_dic[best_pair[1]] == 0:
                vocab_count_dic.pop(best_pair[1])
            splits = merge_pair(*best_pair, word_freq, splits)
            # if i % 10 == 0: # Length of vocabulary for each 10 iterations
            #     print('Vocab len ', len(vocab_count_dic))
            if len(vocab_count_dic) == vocab_size:
                break
            
            i += 1

        vocab = list(vocab_count_dic.keys()) # Creating final vocabulary with <unk>, special and pad tokens.
        vocab.append("<unk>")
        vocab = vocab + special_tokens

        vocab.sort()
        vocab = [pad_token] + vocab
        vocab_dict = {key: i for i, key in enumerate(vocab)}

        Tokenizer = {"vocab_dict": vocab_dict, "merges": merges, "max_iter": max_iterations}

        with open(os.path.join('Saved', f'{tokenizer_file_name}.json'), 'w') as f:
            json.dump(Tokenizer, f)
            print(f"{os.path.join('Saved', f'{tokenizer_file_name}.json')} is successfully created.")  
        
        del vocab
        del vocab_count_dic

        return Tokenizer

    @staticmethod
    def encode(text: str, Tokenizer: dict, eos: str) -> list:
        r"""
        Function to encode the given text using BPE.
        Parameters:
            text (str): Text to be encoded
            Tokenizer (dict): Tokenizer json file that contains vocabulary and merges
            eos (str): eos string
        
        Returns:
            out (list): List of tokens
        """
        vocab_dict = Tokenizer["vocab_dict"]
        merges = Tokenizer["merges"]

        tokens2int = lambda x: vocab_dict.get(x) if x in vocab_dict.keys() else vocab_dict.get('<unk>')
        splits = [" ".join(word.strip()) + " </w>" for word in text.split()]
        splits = [word.split() for word in splits]

        pair_cache = collections.defaultdict(int)
        for split in splits:
            for i in range(len(split) - 1):
                pair_cache[(split[i], split[i+1])] += 1

        for pair, merge in merges.items():
            pair = tuple(pair.split(":"))
            if pair in pair_cache:
                for _, split in enumerate(splits):  
                    i = 0
                    while i < len(split) - 1:
                        if (split[i], split[i + 1]) == pair:
                            split[i:i+2] = [merge]
                        else:
                            i += 1
                    for i in range(len(split) - 1):
                        pair_cache[(split[i], split[i+1])] += 1

        del pair_cache
        if eos != None:
            splits.append([eos])
        return [tokens2int(token) for split in splits for token in split]
    
    @staticmethod
    def decode(tokens: list, Tokenizer: dict) -> str:
        r"""
        Function to decode the given tokens.
        Parameters:
            tokens (list): tokens to be decoded
            Tokenizer (dict): Tokenizer json file that contains vocabulary and merges
        
        Returns:
            out (str): Decoded string
        """
        vocab_dict = Tokenizer["vocab_dict"]
        int2tokens = lambda x: list(vocab_dict.keys())[list(vocab_dict.values()).index(x)]
        out = [int2tokens(c) for c in tokens]
        out = "".join(out)
        out = "".join(out).replace("</w>", " ").strip()
        return out
