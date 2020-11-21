from abc import ABC, abstractmethod
import re

PEGASUS_PATTERN = re.compile("^(\d|[\.,])+$")


class ModelWrapper(ABC):
    """A model wrapper queries a model with a list of text inputs.

    Classification-based models return a list of lists, where each sublist
    represents the model's scores for a given input.

    Text-to-text models return a list of strings, where each string is the
    output – like a translation or summarization – for a given input.
    """

    @abstractmethod
    def __call__(self, text_list):
        raise NotImplementedError()

    @abstractmethod
    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens."""
        raise NotImplementedError()

    def encode(self, inputs):
        """Helper method that calls ``tokenizer.batch_encode`` if possible, and
        if not, falls back to calling ``tokenizer.encode`` for each input.

        Args:
            inputs (list[str]): list of input strings

        Returns:
            tokens (list[list[int]]): List of list of ids
        """
        if hasattr(self.tokenizer, "batch_encode"):
            return self.tokenizer.batch_encode(inputs)
        else:
            return [self.tokenizer.encode(x) for x in inputs]

    def _tokenize(self, inputs):
        """Helper method for `tokenize`"""
        raise NotImplementedError()

    def tokenize(self, inputs, strip_prefix=False, split_num_punct=True):
        """Helper method that tokenizes input strings
        Args:
            inputs (list[str]): list of input strings
            strip_prefix (bool): If `True`, we strip auxiliary characters added to tokens as prefixes (e.g. "##" for BERT, "Ġ" for RoBERTa)
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        tokens = self._tokenize(inputs)
        if strip_prefix:
            # `aux_chars` are known auxiliary characters that are added to tokens
            strip_chars = ["##", "Ġ", "__", "▁"]
            # TODO: Find a better way to identify prefixes. These depend on the model, so cannot be resolved in ModelWrapper.

            def strip(s, chars):
                for c in chars:
                    s = s.replace(c, "")

                return s

            tokens = [[strip(t, strip_chars) for t in x] for x in tokens]

        # with the Pegasus Tokenizer, it seems that numbers that have
        # commas immediately after them are joined with the comma. I.e.
        # a token produced by the PegasusTokenizer might look like: '22,'
        # I think this is a problem; this is fixed with the code below
        if split_num_punct:
            new_tokens = []

            for i in range(len(tokens)):
                new_tokens_i = []
                for token in tokens[i]:
                    if token == '':
                        continue
                    if bool(PEGASUS_PATTERN.match(token)):
                        token_pieces = re.split('(\.|,)', token)
                        for token_piece in token_pieces:
                            if token_piece != '':
                                new_tokens_i.append(token_piece)
                    else:
                        new_tokens_i.append(token)
                new_tokens.append(new_tokens_i)

            tokens = new_tokens

        return tokens
