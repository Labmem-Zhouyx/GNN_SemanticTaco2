# coding:utf-8
import numpy as np
import torch
import os
import argparse
from PIL import Image


from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import dgl
from transformers import BertTokenizer

from hparams import hparams
from models import Semantic_Tacotron2

from distributed import apply_gradient_allreduce
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy

from text import text_to_sequence
from text.dependencyrels import deprel_labels_to_id
from text.DependencyParser_toCharGraph_Stanza import DependencyParser_stanza_word


class TextMelLoaderEval(torch.utils.data.Dataset):
    def __init__(self, sentences, hparams):
        self.sentences = sentences
        self.text_cleaners = hparams.text_cleaners
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


    def text_preprocess(self, text):
        text = text.strip()
        char_seq, char_text_norm = text_to_sequence(text, self.text_cleaners)
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

        edge_node1, edge_node2, edge_label = DependencyParser_stanza_word(char_text_norm)
        edge_type = torch.LongTensor([deprel_labels_to_id[i] for i in edge_label])


        return (char_seq, word_seq, char_word_map, edge_node1, edge_node2, edge_type)


    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.text_preprocess(self.sentences[index])


class TextMelCollateEval():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, hparams):
        self.n_frames_per_step = hparams.n_frames_per_step

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

        return input_lengths, inputs_padded, words_padded, mapping_padded, g_batch


def get_sentences(args):
    if args.text_file != '':
        with open(args.text_file, 'rb') as f:
            sentences = list(map(lambda l: l.decode("utf-8")[:-1], f.readlines()))
    else:
        sentences = [args.sentences]
    print("Check sentences:", sentences)
    return sentences


def load_model(hparams):
    model = Semantic_Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model

def inference(args):

    sentences = get_sentences(args)

    model = load_model(hparams)
    model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    model.cuda().eval()

    test_set = TextMelLoaderEval(sentences, hparams)
    test_collate_fn = TextMelCollateEval(hparams)
    test_sampler = DistributedSampler(valset) if hparams.distributed_run else None
    test_loader = DataLoader(test_set, num_workers=0, sampler=test_sampler, batch_size=hparams.synth_batch_size, pin_memory=False, drop_last=True, collate_fn=test_collate_fn)

    T2_output_range = (-hparams.max_abs_value, hparams.max_abs_value) if hparams.symmetric_mels else (0, hparams.max_abs_value)

    os.makedirs(args.out_filename, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            mel_outputs, mel_outputs_postnet, _, alignments = model.inference(batch)
            align_img = Image.fromarray(plot_alignment_to_numpy(alignments[0].data.cpu().numpy().T))
            spec_img = Image.fromarray(plot_spectrogram_to_numpy(mel_outputs_postnet[0].data.cpu().numpy()))
            align_img.save(os.path.join(args.out_filename, 'sentence_{}_alignment.jpg'.format(i)))
            spec_img.save(os.path.join(args.out_filename, 'sentence_{}_mel-spectrogram.jpg'.format(i)))
            mels = mel_outputs_postnet[0].cpu().numpy()

            mel_path = os.path.join(args.out_filename, 'sentence_{}_mel-feats.npy'.format(i))
            mels = np.clip(mels, T2_output_range[0], T2_output_range[1])
            np.save(mel_path, mels.T, allow_pickle=False)

            print('CHECK MEL SHAPE:', mels.T.shape)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--sentences', type=str, help='text to infer', default='Hello.')
    parser.add_argument('-t', '--text_file', type=str, help='text file to infer', default='./sentences_en.txt')
    parser.add_argument('-c', '--checkpoint', type=str, help='checkpoint path',
                       default='./training/train_bitype_semantic_Taco2_optAdam_gated_5_b16_ljspeech/checkpoint_150000')

    parser.add_argument('-o', '--out_filename', type=str, help='output filename', default='./inference_mel')
    args = parser.parse_args()
    inference(args)