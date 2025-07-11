# -*- coding: utf-8 -*-
import itertools
from typing import List, Sequence, Tuple
import torch

# from https://github.com/facebookresearch/esm/blob/main/esm/constants.py
proteinseq_toks = {
    "toks": [
        "L",
        "A",
        "G",
        "V",
        "S",
        "E",
        "R",
        "T",
        "I",
        "D",
        "P",
        "K",
        "Q",
        "N",
        "F",
        "Y",
        "M",
        "H",
        "W",
        "C",
        "X",
        "B",
        "U",
        "Z",
        "O",
        ".",
        "-",
    ],
}

# from https://github.com/facebookresearch/esm/blob/main/esm/data.py
class Alphabet(object):
    def __init__(
        self,
        standard_toks: Sequence[str] = proteinseq_toks["toks"],
        prepend_toks: Sequence[str] = ("<cls>", "<pad>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<mask>",),
        prepend_bos: bool = True,
        append_eos: bool = True,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")
        self.all_special_tokens = ["<eos>", "<unk>", "<pad>", "<cls>", "<mask>"]
        self.unique_no_split_tokens = self.all_toks

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:
        """
        Inspired by https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
        Converts a string in a sequence of tokens, using the tokenizer.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.

        Returns:
            :obj:`List[str]`: The list of tokens.
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # AddedToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # We strip left and right by default
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def untokenize(self, indices):
        if torch.is_tensor(indices):
            return "".join([self.get_tok(int(i.item())) for i in indices])
        else:
            return "".join([self.get_tok(i) for i in indices])


    # def encode(self, text):
    #     return [self.tok_to_idx[tok] for tok in self.tokenize(text)]

    # def _feat_text(self, text, feat_id):
    #     assert feat_id < 4
    #     return [self.idx_prop_feat[idx][feat_id] for idx in self.encode(text)]

    # def _feat_idx(self, indices, feat_id):
    #     assert feat_id < 4
    #     return [self.idx_prop_feat[idx][feat_id] for idx in indices]

    # def feat_text(self, text):
    #     return {
    #         "chem_polar": self._feat_text(text, 0),
    #         "net_charge": self._feat_text(text, 1),
    #         "hydropathy": self._feat_text(text, 2),
    #         "mol_mass": self._feat_text(text, 3),
    #     }

    # def feat_idx(self, indices):
    #     return {
    #         "chem_polar": self._feat_idx(indices, 0),
    #         "net_charge": self._feat_idx(indices, 1),
    #         "hydropathy": self._feat_idx(indices, 2),
    #         "mol_mass": self._feat_idx(indices, 3),
    #     }

def convert_string_sequence_to_int_index(vocab, seq):
    seq_sep = vocab.tokenize(seq)
    tokens = [vocab.tok_to_idx[tok] for tok in seq_sep]
    if vocab.prepend_bos:
        tokens.insert(0, vocab.cls_idx)
    if vocab.append_eos:
        tokens.append(vocab.eos_idx)
    
    return tokens


if __name__ == "__main__":
    alphabet = Alphabet()
    print("Size", len(alphabet))
    print(alphabet.all_toks)
    print(len(alphabet.all_toks))
    # print(alphabet.tok_to_idx)
    protein_seq = "AAA<mask>LM<mask>MLMLM<mask>A<mask>"
    tokens = alphabet.tokenize(protein_seq)
    print(tokens)    
    tokens = [alphabet.tok_to_idx[tok] for tok in tokens]
    print(tokens)
    print(alphabet.mask_idx)
    print(alphabet.padding_idx)
    # print()
    # print(alphabet.feat_text("AAAALMLMLMLM<mask>AAA"))
    # print(alphabet.feat_idx(alphabet.encode("AAAALMLMLMLM<mask>AAA")))
