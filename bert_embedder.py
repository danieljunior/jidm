# import logging

from transformers import BertModel, BertTokenizer
import torch
import nltk

nltk.download('punkt')


class BertEmbedder:
    MAX_LENGTH = 512

    def __init__(self, model_path):
        self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
        self.model = BertModel.from_pretrained(model_path, output_hidden_states=True)

    def get_embeddings(self, text):
        outputs = self.get_features(text)
        embeddings = []
        for last_hidden_state, pooler_output, hidden_states in outputs:
            embeddings.append(BertEmbedder.get_concat_four_last_layers(hidden_states))

        return torch.mean(torch.stack(embeddings), dim=0)

    def get_features(self, text):
        text_split = self.get_split(text, max_length=BertEmbedder.MAX_LENGTH)
        resp = []
        for split in text_split:
            input_ids, segment_ids, input_mask = BertEmbedder.convert_examples_to_features(
                example=split, seq_length=BertEmbedder.MAX_LENGTH, tokenizer=self.tokenizer)
            with torch.no_grad():
                outputs = self.model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
                # last_hidden_state, pooler_output, hidden_states
            resp.append(outputs)
        return resp

    @staticmethod
    def convert_examples_to_features(example, seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""
        tokens = ['[CLS]']
        for i, w in enumerate(nltk.tokenize.word_tokenize(example, language='portuguese')):
            sub_words = tokenizer.tokenize(w)
            if not sub_words:
                sub_words = ['[UNK]']
            tokens.extend(sub_words)

        # truncate
        if len(tokens) > seq_length - 1:
            # logger.info('Example is too long, length is {}, truncated to {}!'.format(
            #     len(tokens), seq_length))
            tokens = tokens[0:(seq_length - 1)]
        tokens.append('[SEP]')

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        input_ids = torch.tensor([input_ids], dtype=torch.long)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long)
        input_mask = torch.tensor([input_mask], dtype=torch.long)

        return input_ids, segment_ids, input_mask

    @staticmethod
    def get_last_layer(hidden_states):
        return hidden_states[-1]

    @staticmethod
    def get_summed_four_last_layers(hidden_states):
        return torch.stack(hidden_states[-4:]).sum(0)

    @staticmethod
    def get_concat_four_last_layers(hidden_states):
        pooled_output = torch.cat(tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)
        return pooled_output[:, 0, :]

    @staticmethod
    def ceiling_division(n, d):
        return -(n // -d)

    @staticmethod
    def get_split(text, max_length=200, overlap=50):
        l_total = []
        l_parcial = []
        text_len = len(text.split())
        aux_value = (max_length - overlap)
        splits = BertEmbedder.ceiling_division(text_len, aux_value)
        if splits > 0:
            n = splits
        else:
            n = 1
        for w in range(n):
            if w == 0:
                l_parcial = text.split()[:max_length]
                l_total.append(" ".join(l_parcial))
            else:
                l_parcial = text.split()[w * aux_value:w * aux_value + max_length]
                l_total.append(" ".join(l_parcial))
        return l_total
