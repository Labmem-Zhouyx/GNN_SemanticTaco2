import json
import random
import codecs
import numpy as np
import torch
import torch.utils.data

from utils import load_filepaths_and_text

from transformers import BertTokenizer
from text import text_to_sequence
from text.dependencyrels import deprel_labels_to_id
from text.DependencyParser_toCharGraph_Stanza import DependencyParser_stanza_word

import dgl

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, melpaths_and_text, hparams):
        self.melpaths_and_text = load_filepaths_and_text(melpaths_and_text)
        self.text_cleaners = hparams.text_cleaners

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        random.seed(hparams.seed)
        random.shuffle(self.melpaths_and_text)

    def get_mel_text_pair(self, melpath_and_text):
        # separate filename and text
        melpath, char_text, node1_text, node2_text, deprel_text = melpath_and_text[0], melpath_and_text[1], melpath_and_text[2], melpath_and_text[3], melpath_and_text[4]
        char_seq, word_seq, char_word_map = self.text_preprocess(char_text)
        mel = torch.from_numpy(np.load(melpath))
        edge_node1 = [int(i) for i in node1_text.split(' ')]
        edge_node2 = [int(i) for i in node2_text.split(' ')]
        edge_type = torch.LongTensor([deprel_labels_to_id[i] for i in deprel_text.split(' ')])
        if max(edge_node1) >= len(word_seq):
            print(char_text)
            print(edge_node1)
            print(word_seq)
        return (char_seq, word_seq, char_word_map, edge_node1, edge_node2, edge_type, mel)

    def text_preprocess(self, char_text):
        # dealing with BERT tokenizer output:
        # the length of output doesn't match the length of words,
        # because some words are divided to a few tokens. eg: "subword" becomes "sub" and "##word"
        char_seq, char_text_norm = text_to_sequence(char_text, self.text_cleaners)
        char_seq = torch.LongTensor(char_seq)
        toks_raw = self.tokenizer.tokenize(char_text_norm)
        toks = [i for i in toks_raw if not i.startswith('##')]
        word_seq = torch.tensor(self.tokenizer.convert_tokens_to_ids(toks), dtype=torch.long)

        char_word_map = []
        index = -1
        counter = 0
        for tok in toks_raw:
            counter += len(tok)
            if tok.startswith('##'):
                counter -= 2
                for j in range(len(tok) - 2):
                    char_word_map.append(index)
            else:
                index += 1
                for j in range(len(tok)):
                    char_word_map.append(index)

            if counter < len(char_text_norm) and char_text_norm[counter] == ' ':
                char_word_map.append(index)
                counter += 1

        assert len(char_seq) == len(char_word_map)
        char_word_map = torch.LongTensor(char_word_map)

        return char_seq, word_seq, char_word_map

    def __len__(self):
        return len(self.melpaths_and_text)

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.melpaths_and_text[index])


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        inputs_padded = torch.LongTensor(len(batch), max_input_len)
        inputs_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            input_id = batch[ids_sorted_decreasing[i]][0]
            inputs_padded[i, :input_id.shape[0]] = input_id

        max_words_len = max(len(x[1]) for x in batch)
        words_padded = torch.LongTensor(len(batch), max_words_len)
        words_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            words_id = batch[ids_sorted_decreasing[i]][1]
            words_padded[i, :len(words_id)] = words_id

        mapping_padded = torch.LongTensor(len(batch), max_input_len)
        mapping_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            mapping_id = batch[ids_sorted_decreasing[i]][2] + max_words_len * i
            mapping_padded[i, :mapping_id.shape[0]] = mapping_id

        g_list = []
        for i in range(len(ids_sorted_decreasing)):
            g = dgl.graph((batch[ids_sorted_decreasing[i]][3], batch[ids_sorted_decreasing[i]][4]),
                          num_nodes=max_words_len)
            g.edata['type'] = batch[ids_sorted_decreasing[i]][5]
            g_list.append(g)
        g_batch = dgl.batch(g_list)

        # Right zero-pad mel-spec
        num_mels = batch[0][6].size(0)
        max_target_len = max([x[6].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][6]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return input_lengths, inputs_padded, words_padded, mapping_padded, g_batch, mel_padded, gate_padded, output_lengths