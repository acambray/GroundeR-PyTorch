"""
Author: Aleix Cambray Roma
Work done while at Imperial College London.
"""

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import time
import copy
import pickle
import numpy as np
from model import GroundeR
import matplotlib.pyplot as plt
from dataloader_unsupervised import UnsupervisedDataLoader


def print_gpu_memory(first_string=None):
    if first_string is not None:
        print(first_string)
    print('   current memory allocated: {}'.format(torch.cuda.memory_allocated() / 1024 ** 2))
    print('   max memory allocated: {}'.format(torch.cuda.max_memory_allocated() / 1024 ** 2))
    print('   cached memory: {}'.format(torch.cuda.memory_cached() / 1024 ** 2))


def seq2phrase(seq):
    return " ".join([idx2word[idx.item()] for idx in seq])


if __name__ == '__main__':
    print("Running on {}".format(torch.cuda.get_device_name(0)))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print_gpu_memory()

    # Load vocabulary and embeddings
    # vocab_file = "C:/Data/GroundeR/vocabulary.pkl"
    vocab_file = "C:/Data/Embeddings/glove.6B/GloVe_6B_50d_vocabulary.pkl"
    with open(vocab_file, "rb") as f:
        vocab, word2idx, idx2word, weight_matrix = pickle.load(f)
        vocab = list(word2idx.keys())

    # Config
    vocab_size = len(vocab)
    embed_dim = 50
    h_dim = 100
    v_dim = 1000

    # Hyper-parameters
    regions = 25
    learning_rate = 0.00025
    epochs = 100
    batch_size = 30
    max_seq_length = 10
    L = 5

    # Build data pipeline providers
    dataset = UnsupervisedDataLoader(data_dir="C:/Data/GroundeR/", sample_list_file="flickr30k_train_val.txt", seq_length=max_seq_length)
    dataset_length = len(dataset)

    train_length = int(0.8*dataset_length)
    val_length = dataset_length - train_length
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_length, val_length])

    print("Training dataset size: {}".format(len(train_ds)))
    print("Val dataset size:      {}".format(len(val_ds)))
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3)

    # Instantiate model and send to GPU if available
    model = GroundeR(vocab, vocab_size, embed_dim, h_dim, v_dim, regions, weight_matrix, train_embeddings=False)
    model = model.to(device)
    print_gpu_memory("  After model to GPU:")
    criterion_att = nn.NLLLoss()
    criterion_rec = nn.NLLLoss(ignore_index=0)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    t0 = time.time()
    start = time.time()
    min_loss = 10000
    times = []
    times_total = []
    training_losses = []
    training_accuracies = []
    epoch_training_losses = []
    epoch_training_accuracies = []
    epoch_val_losses = []
    epoch_val_accuracies = []
    iter_num = []
    for epoch in range(epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                model.train()  # Set model to training mode
            else:
                dataloader = val_dataloader
                model.eval()   # Set model to evaluation mode

            epoch_samples = 0
            running_loss = 0.0
            running_corrects = 0.0
            batch_i = 0
            for vis_features, real_feat, encoder_input, decoder_input, decoder_target, mask, region_true, lengths_enc, lengths_dec in dataloader:
                torch.cuda.empty_cache()
                dt_load = (time.time() - t0) * 1000

                # Send all data to GPU if available
                t1 = time.time()
                vis_features = vis_features.to(device)
                real_feat = real_feat.to(device)
                encoder_input = encoder_input.to(device)
                decoder_input = decoder_input.to(device)
                decoder_target = decoder_target.to(device)
                mask = mask.to(device)
                region_true = region_true.to(device)
                # lengths_enc = lengths_enc.to(device)
                # lengths_dec = lengths_dec.to(device)
                dt_togpu = (time.time() - t1) * 1000
                print_gpu_memory("  After data to GPU:")

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward Pass
                t2 = time.time()
                decoder_output, att, att_log = model(encoder_input, lengths_enc, vis_features, decoder_input, lengths_dec)
                dt_forward = (time.time() - t2) * 1000
                print_gpu_memory("  After forward run:")

                # Loss
                # TODO: [done] Mask loss to ignore pad tokens (using the mask tensor for each sample)
                t3 = time.time()
                pred = decoder_output.view(decoder_output.size(0)*decoder_output.size(1), decoder_output.size(2))
                target = decoder_target[:, :decoder_output.size(1)]  # truncate the decoder_target from length 10 to maximum sequence length in batch
                target = target.contiguous().view(-1)
                loss_att = criterion_att(att_log, region_true)
                # loss_rec = model.masked_NLLLoss(decoder_output, decoder_target)
                loss_rec = criterion_rec(pred, target)
                loss = L*loss_att + loss_rec
                dt_loss = (time.time() - t3) * 1000
                print_gpu_memory("  After loss calc:")

                # Accuracy
                region_pred = att.max(dim=1)[1]
                corrects = torch.sum(region_true == region_pred).item()
                running_corrects += corrects
                accuracy = corrects / region_true.size(0)

                if phase == 'train':
                    # Backward pass and parameter update
                    # TODO: [done] Find out why loss isn't getting updates (initialisation?)
                    t4 = time.time()
                    loss.backward()
                    print_gpu_memory("  After backward run:")
                    optimizer.step()
                    print_gpu_memory("  After optimizer step:")
                    dt_backward = (time.time() - t4) * 1000
                    print("{:02.0f}.{:03.0f} - Sample {:05.0f}/30781 - Accuracy: {:0.2f}% - Loss: {:02.7f} - GPU: {:0.2f} MB - Load time = {:06.2f}ms - toGPU time = {:06.2f}ms - Forward time = {:06.2f}ms - Loss: {:06.2f}ms - Backward {:06.2f}ms - Time {:05.2f}s".format(epoch, batch_i + 1, (batch_i + 1) * batch_size, accuracy*100, loss.item(), torch.cuda.memory_allocated() / 1024 ** 2, dt_load, dt_togpu, dt_forward, dt_loss, dt_backward, (time.time()-start)))
                    training_losses.append(loss.item())  # Appends loss over entire batch (reduce=mean)
                    training_accuracies.append(accuracy)
                    times.append(dt_load + dt_togpu)
                    times_total.append(time.time() - t0)

                # statistics & counters
                epoch_samples += vis_features.size(0)
                running_loss += loss.item() * vis_features.size(0)            # Sum of losses over all samples in batch

                batch_i += 1
                t0 = time.time()

                # Print real and predicted sentences:
                for i_print in range(3):
                    print("   {} - Real: {}".format(i_print, seq2phrase(decoder_target[i_print])))
                    print("   {} - Pred: {}".format(i_print, seq2phrase(decoder_output.max(dim=2)[1][i_print])))

            # Track epoch losses for both training and validation phases
            if phase == 'train':
                print("Training Epoch Performance on {} samples".format(epoch_samples))
                epoch_loss = running_loss / epoch_samples
                epoch_training_losses.append(epoch_loss)
                epoch_training_accuracies.append(running_corrects/epoch_samples)
                iter_num.append(len(training_losses))
            elif phase == 'val':
                print("Validation Performance on {} samples".format(epoch_samples))
                epoch_loss = running_loss / epoch_samples
                epoch_val_losses.append(epoch_loss)
                epoch_val_accuracies.append(running_corrects/epoch_samples)
                # Save weights of best val-performing model
                if epoch_loss < min_loss:
                    min_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), "best_model.pt")

            # Print learning profile every 5 epochs
            if phase == 'val':
                plt.title("LR: {}".format(learning_rate))
                plt.plot(training_losses, label="Training Losses")
                plt.plot(iter_num, epoch_training_losses, marker='o', label="Epoch Training losses")
                plt.plot(iter_num, epoch_val_losses, marker='o', label="Epoch Val Losses")
                plt.legend()
                plt.savefig('learning_profile.png')
                plt.clf()
                plt.plot(training_accuracies, label="Training Accuracies per batch")
                plt.plot(iter_num, epoch_training_accuracies, label="Epoch Training Accuracies")
                plt.plot(iter_num, epoch_val_accuracies, label="Epoch Val Accuracies")
                plt.title("Region Accuracies")
                plt.savefig('accuracies.png')
                plt.clf()

    plt.plot(times)
    plt.savefig('final_times.png')

    plt.plot(times_total)
    plt.savefig('final_times_total.png')

    plt.plot(training_losses, label="Training Losses")
    plt.plot(iter_num, epoch_training_losses, label="Epoch Training losses")
    plt.plot(iter_num, epoch_val_losses, label="Epoch Val Losses")
    plt.legend()
    plt.savefig('final_learning_profile.png')

    print("\nTime per sample: {}".format((time.time()-t0)/30781))