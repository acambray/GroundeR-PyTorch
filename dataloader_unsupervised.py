"""
Author: Aleix Cambray Roma
Work done while at Imperial College London.
"""
import torch
import pickle
import numpy as np
from numpy.random import randint
from torch.utils.data import Dataset


class UnsupervisedDataLoader(Dataset):
    """ Loads Flickr30k data for the unsupervised, reconstruction case. """
    def __init__(self, data_dir, sample_list_file, seq_length):
        def read_sample_list(sample_list_file):
            f = open(data_dir + sample_list_file)
            return np.array([sample_id.strip() for sample_id in f.readlines()])

        self.seq_length = seq_length
        self.sample_id_list = read_sample_list(sample_list_file)
        self.data_dir = data_dir

        with open(self.data_dir + "vocabulary.pkl", "rb") as f:
            self.word2idx, self.idx2word = pickle.load(f)
            self.vocab = list(self.word2idx.keys())

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, i):
        sample_id = self.sample_id_list[i]

        # Get Visual feature matrix
        vis_features = np.zeros((25, 1000), dtype='float32')
        real_feat = np.load(self.data_dir + "visualfeatures_data/" + sample_id + ".npy")
        vis_features[:real_feat.shape[0], :] = real_feat
        real_feat = real_feat.shape[0]

        # Get annotations (all phrases in this image)
        with open(self.data_dir + "annotation_data/" + sample_id + ".pkl", "rb") as f:
            annotations = pickle.load(f)
        phrases = annotations['seqs']

        # Select random phrase from this image
        # print(len(phrases))
        if len(phrases) == 0:
            print(sample_id)
            print(real_feat)
            print("")

        rand_int = randint(len(phrases))
        phrase = phrases[rand_int]

        if len(phrase) > self.seq_length:
            phrase[self.seq_length-1] = phrase[-1]               # Add end tag at last step
            phrase = phrase[:self.seq_length]                    # Truncate phrase to seq_length

        # Initialise all phrase arrays with the padding symbol
        pad_idx = self.word2idx['<pad>']
        encoder_input = np.ones(self.seq_length, dtype='int64')*pad_idx
        decoder_input = np.ones(self.seq_length, dtype='int64')*pad_idx
        decoder_target = np.ones(self.seq_length, dtype='int64')*pad_idx
        mask = np.zeros(self.seq_length, dtype='int64')

        # Replace the first padding symbols by the real phrase
        encoder_input[0:len(phrase)] = np.array(phrase)          # Feed to encoder LSTM: Both <start> or <end> tags
        decoder_input[0:len(phrase)-2] = np.array(phrase[1:-1])  # Feed to decoder LSTM: No tags
        decoder_target[0:len(phrase)-1] = np.array(phrase[1:])   # Used as reconstruction ground truth: Only <end> tag
        mask[0:len(phrase)-1] = 1

        with open('C:/Data/GroundeR/id2idx_data/'+sample_id+'.pkl', 'rb') as f:
            id2idx = pickle.load(f)

        phrase_id = annotations['ids'][rand_int]
        true_region = id2idx[phrase_id]
        # Sentence as list of string-words (visualise)
        # phrase_word = np.array([self.idx2word[idx] for i, idx in enumerate(encoder_input) if i < len(phrase)])

        # print("Shapes:")
        # print(" - vis_features: {}".format(vis_features.shape))
        # print(" - encoder_input: {}".format(encoder_input.shape))
        # print(" - decoder_input: {}".format(decoder_input.shape))
        # print(" - decoder_target: {}".format(decoder_target.shape))

        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # print('current memory allocated: {}'.format(torch.cuda.memory_allocated() / 1024 ** 2))
        # print('max memory allocated: {}'.format(torch.cuda.max_memory_allocated() / 1024 ** 2))
        # print('cached memory: {}'.format(torch.cuda.memory_cached() / 1024 ** 2))

        # vis_features = torch.from_numpy(vis_features)
        # real_feat = torch.tensor(real_feat).to(device)
        # encoder_input = torch.from_numpy(encoder_input)
        # decoder_input = torch.from_numpy(decoder_input)
        # decoder_target = torch.from_numpy(decoder_target)
        # mask = torch.from_numpy(mask)

        return vis_features, real_feat, encoder_input, decoder_input, decoder_target, mask, true_region, len(phrase), len(phrase)-1
