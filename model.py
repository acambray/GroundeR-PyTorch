"""
Author: Aleix Cambray Roma
Work done while at Imperial College London.
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PhraseEncoder(nn.Module):
    """
    Phrase Encoder
    Input is sequence of word indices
    Output is last LSTM hidden state
    """
    def __init__(self, embedding_layer, vocab_size, embed_dim, h_dim):
        super(PhraseEncoder, self).__init__()

        # Word embedding layers
        # self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding = embedding_layer

        # LSTM layer
        self.lstm = nn.LSTM(embed_dim, h_dim, batch_first=True)

    def forward(self, phrase_batch, lengths_enc):
        """
        Embedding turns sequences of indices (phrase) into sequence of vectors
        :param phrase_batch: (BATCH, TIME) i.e. if batch_size=2 and seq_length=5 phrase=[[1,5,4,7,3], [8,2,5,2,4]]
        :return: last LSTM output
        """

        #   phrase dimensions: (BATCH, TIME) i.e. if batch_size=2 and seq_length=5 phrase=[[1,5,4,7,3], [8,2,5,2,4]]
        #   embeds dimensions:

        batch_size = phrase_batch.size()[0]
        seq_length = phrase_batch.size()[1]
        embeds = self.embedding(phrase_batch)

        # Pack
        #  1. sort sequences by length
        ordered_len, ordered_idx = lengths_enc.sort(0, descending=True)
        ordered_embeds = embeds[ordered_idx]
        #  2. pack
        input_packed = pack_padded_sequence(ordered_embeds, ordered_len, batch_first=True)

        # Feed embeddings to LSTM
        # embeds = embeds.view(seq_length, batch_size, -1)
        _, (ht, ct) = self.lstm(input_packed)  # Hidden is none because we don't initialise the hidden state, could use random noise instead

        # Get final hidden state from LSTM and reverse descending ordering
        h = ht[0, :, :]
        h[ordered_idx] = h
        return h


class AttentionModule(nn.Module):
    def __init__(self, h_dim, v_dim, embed_dim, out1, regions):
        super(AttentionModule, self).__init__()
        self.fc1 = nn.Linear(h_dim + v_dim, out1, bias=True)
        self.fc2 = nn.Linear(out1, 1, bias=True)
        self.fcREC = nn.Linear(v_dim, embed_dim, bias=True)

    def forward(self, h, v):
        """
        :param h: encoded phrases            dimensions [batch, h_dim]                 [32, 100]
        :param v: visual features matrix     dimensions [batch_size, regions, v_dim]   [32,  25, 1000]
        :return:
        """
        batch_size = h.size()[0]
        regions = v.size()[1]
        h_dim = h.size()[1]
        v_dim = v.size()[2]

        # We want to turn h from [batch_size, h_dim] to [batch_size, regions, 1, h_dim]
        h1 = h[:, None, None, :]                                                    # shape: (batch_size,       1, 1, h_dim)
        h_reshaped = h1.repeat(1, regions, 1, 1).type(dtype=torch.float32)          # shape: (batch_size, regions, 1, h_dim)

        # Add extra dimension to match
        v_reshaped = v[:, :, None, :].type(dtype=torch.float32)                     # shape: (batch_size, regions, 1, v_dim)

        hv = torch.cat((v_reshaped, h_reshaped), dim=3)

        # View hv (batch_size, regions, 1, hv_dim) to (batch_size*regions, hv_dim)
        x = hv.view((hv.size(0)*hv.size(1), hv.size(3)))
        x = self.fc1(x)                                                             # [ batch*reg, hidden ]
        x = F.relu(x)
        x = self.fc2(x)                                                             # [ batch*reg,      1 ]
        x = x.view(hv.size(0), hv.size(1))                                          # [     batch,    reg ]
        alpha = F.softmax(x, dim=1)                                                 # [     batch,    reg ]
        att_log = F.log_softmax(x, dim=1)

        # Sum of the elementwise product between alpha and v
        alpha_expanded = alpha[:, :, None].expand_as(v)
        v_att = torch.sum(torch.mul(v, alpha_expanded), dim=(1,))

        v_att_dec = self.fcREC(v_att)
        v_att_dec = F.relu(v_att_dec)
        return v_att, v_att_dec, alpha, att_log


class PhraseDecoder(nn.Module):
    def __init__(self, embedding_layer, embed_dim, h_dim, vocab_size, v_dim):
        super(PhraseDecoder, self).__init__()

        self.embedding = embedding_layer

        self.decoder = nn.LSTM(embed_dim, h_dim, batch_first=True)
        self.hidden_to_prob = nn.Linear(h_dim, vocab_size)

    def forward(self, v_att_dec, decoder_input, lengths_dec, teacher_forcing=1):
        # TODO: Implement greedy strategy (using 'teacher forcing' at the moment)

        embeds = self.embedding(decoder_input)

        # Concatenate the visual attended context vector and the phrase embedding as the input to the decoder
        decoder_input = torch.cat((v_att_dec[:, None, :], embeds[:, :-1, :]), dim=1)                     # indexes: [batch, timestep, word]

        # Re-order axis to fit PyTorch LSTM convention (seq_length, batch, input_size)
        # decoder_input = decoder_input.permute((1, 0, 2))

        # Pack --------------------------------------------------------------------------------------
        #  1. sort sequences by length
        ordered_len, ordered_idx = lengths_dec.sort(0, descending=True)
        ordered_embeds = decoder_input[ordered_idx]
        #  2. pack
        input_packed = pack_padded_sequence(ordered_embeds, ordered_len, batch_first=True)

        # LSTM --------------------------------------------------------------------------------------
        output_packed, hidden = self.decoder(input_packed)

        # Unpack all hidden states-------------------------------------------------------------------
        output_sorted, _ = pad_packed_sequence(output_packed, batch_first=True)

        # Fully Connected (hidden state to vocabulary probabilities) --------------------------------
        output = torch.zeros_like(output_sorted)
        output[ordered_idx] = output_sorted                                                         # shape: batch_size  x  seq_len  x  hidden

        input_fc = output.contiguous().view(output.size(0)*output.size(1), output.size(2))          # shape: batch_size*seq_len   x  hidden
        out = self.hidden_to_prob(input_fc)                                                         # shape: batch_size*seq_len   x  vocab_size
        out = out.view(output.size(0), output.size(1), out.size(1))                                 # shape: batch_size  x   seq_len   x  vocab_size]
        decoder_output = F.log_softmax(out, dim=2)

        return decoder_output


class GroundeR(nn.Module):
    def __init__(self, vocab, vocab_size, embed_dim, h_dim, v_dim, regions, embeddings_matrix=None, train_embeddings=False):
        super(GroundeR, self).__init__()

        if embeddings_matrix is None:
            self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=False)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddings_matrix).type(torch.float32), freeze=True, sparse=True)
            if train_embeddings is False:
                self.embedding.weight.requires_grad = False

        self.phrase_encoder = PhraseEncoder(self.embedding, vocab_size, embed_dim, h_dim)
        self.attention = AttentionModule(h_dim, v_dim, embed_dim, 100, regions)
        self.phrase_decoder = PhraseDecoder(self.embedding, embed_dim, h_dim, vocab_size, v_dim)

        self.vocab = vocab
        self.vocab_size = len(vocab)

    def forward(self, encoder_input, lengths_enc, vis_features, decoder_input, lengths_dec):

        # Encode
        encoded_batch = self.phrase_encoder(encoder_input, lengths_enc)

        # Attend
        v_att, v_att_dec, att, att_log = self.attention(encoded_batch, vis_features)

        # Decode
        decoder_output = self.phrase_decoder(v_att_dec, decoder_input, lengths_dec, teacher_forcing=1)

        return decoder_output, att, att_log

    def masked_NLLLoss(self, pred, target, pad_token=0):
        max_pred_len = pred.size(1)
        Y = target[:, :max_pred_len]
        Y = Y.contiguous().view(-1)

        Y_hat = pred.view(-1, pred.size(2))

        mask = (Y != pad_token).float()
        Y_hat_masked = torch.zeros_like(Y, dtype=torch.float32)

        for i, idx in enumerate(Y):
            Y_hat_masked[i] = Y_hat[i, idx]

        Y_hat_masked = Y_hat_masked * mask

        n_tokens = torch.sum(mask)

        loss = -torch.sum(Y_hat_masked) / n_tokens
        return loss
