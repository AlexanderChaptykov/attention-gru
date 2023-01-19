from __future__ import unicode_literals, print_function, division

import random
import time
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from seq2seq_attn.constants import device, teacher_forcing_ratio, PRINT_EVERY
from seq2seq_attn.data_prep import MAX_LENGTH
from seq2seq_attn.dataset import TwoLangs, SOS_token, tensorsFromPair
from seq2seq_attn.models import AttnDecoderRNN, EncoderRNN
from seq2seq_attn.utils import timeSince

plt.switch_backend("agg")


def teacher_force(
    batch, decoder, decoder_input, decoder_hidden, encoder_outputs, criterion
):
    loss = 0
    for di in range(batch.dec.shape[1] - 1):
        decoder_output, decoder_hidden, attn_weights = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        decoder_input = batch.dec[:, di : di + 1]  # Teacher forcing
        loss += criterion(decoder_output, batch.dec[:, di])
    return loss


def no_teacher_force(
    batch, decoder, decoder_input, decoder_hidden, encoder_outputs, criterion
):
    loss = 0
    for di in range(batch.dec.shape[1] - 1):
        decoder_output, decoder_hidden, attn_weights = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.detach()  # detach from history as input # torch.Size([2])
        loss += criterion(decoder_output, batch.dec[:, di])
    return loss


def batch_train(
    batch: TwoLangs,
    encoder: EncoderRNN,
    decoder: AttnDecoderRNN,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_outputs, encoder_hidden = encoder(batch.enc)

    decoder_input = torch.tensor([[SOS_token]] * len(batch.enc), device=device)
    decoder_hidden = encoder_hidden

    if random.random() < teacher_forcing_ratio:
        loss = teacher_force(
            batch, decoder, decoder_input, decoder_hidden, encoder_outputs, criterion
        )
    else:
        loss = no_teacher_force(
            batch, decoder, decoder_input, decoder_hidden, encoder_outputs, criterion
        )

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / len(batch.dec)


def collate_fn(batch: List[TwoLangs], max_len=MAX_LENGTH):
    enc = [x.enc.squeeze(1) for x in batch]
    dec = [x.dec.squeeze(1) for x in batch]
    if max_len is not None:
        enc[0] = nn.ConstantPad1d((0, max_len - enc[0].shape[0]), 0)(enc[0])
        dec[0] = nn.ConstantPad1d((0, max_len - dec[0].shape[0]), 0)(dec[0])

    enc = pad_sequence(enc, batch_first=True)
    dec = pad_sequence(dec, batch_first=True)
    return TwoLangs(enc=enc, dec=dec)


def log_resuls(iteration, print_loss_total, print_every, start, n_iters):
    if iteration % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print(
            "%s (%d %d%%) %.4f"
            % (
                timeSince(start, iteration / n_iters),
                iteration,
                iteration / n_iters * 100,
                print_loss_avg,
            )
        )
        print_loss_total = 0
    # if iter % plot_every == 0:
    #     plot_loss_avg = plot_loss_total / plot_every
    #     plot_losses.append(plot_loss_avg)
    #     plot_loss_total = 0


def train(
    input_lang,
    output_lang,
    pairs,
    encoder,
    decoder,
    n_iters,
    batch_size=32,
    n_epochs=1,
    plot_every=100,
    learning_rate=0.01,
    print_every=PRINT_EVERY,
):
    start = time.time()
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    criterion = nn.NLLLoss()

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # dataset = CustomDs(input_lang, output_lang, pairs[:n_iters])
    for i in range(n_epochs):
        print("epoch", i)
        training_pairs = [
            tensorsFromPair(input_lang, output_lang, random.choice(pairs))
            for _ in range(n_iters)
        ]
        dl = DataLoader(
            training_pairs, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
        for iteration, batch in enumerate(dl):
            loss = batch_train(
                batch, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion
            )
            print_loss_total += loss
            plot_loss_total += loss
            if (iteration + 1) % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print(
                    "%s (%d %d%%) %.4f"
                    % (
                        timeSince(start, iteration / n_iters),
                        iteration,
                        iteration / n_iters * 100,
                        print_loss_avg,
                    )
                )
                print_loss_total = 0
    return
