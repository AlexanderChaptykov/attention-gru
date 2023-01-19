import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq_attn.constants import device
from seq2seq_attn.data_prep import MAX_LENGTH
from seq2seq_attn.dataset import SOS_token, EOS_token


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=2, bidirect=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirect,
        )

    def forward(self, input):
        """

        Parameters
        ----------
        input: tensor of tokens shape [bs, max_len]

        Returns
        -------
        output: encoded tokens shape [bs, max_len, hidden_size]
        """
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, self.init_hidden(bs=input.shape[0]))
        return output, hidden

    def init_hidden(self, bs):
        bid_coef = 2 if self.gru.bidirectional else 1
        return torch.zeros(
            bid_coef * self.gru.num_layers, bs, self.gru.hidden_size, device=device
        )


class Attention(nn.Module):
    def __init__(self, hidden_size, max_length=MAX_LENGTH, bidirect=False):
        super().__init__()
        torch.manual_seed(0)
        bid_coef = 3 if bidirect else 2
        self.attn = nn.Linear(hidden_size * bid_coef, max_length)

        torch.manual_seed(0)
        self.attn_combine = nn.Linear(hidden_size * bid_coef, hidden_size)

    def forward(self, embedded_input, hidden, encoder_outputs):
        if self.is_bidirect(hidden):
            hidden = hidden.transpose(0, 1).flatten(1, 2)
        else:
            hidden = hidden.squeeze(0)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded_input, hidden), -1)), dim=1
        ).unsqueeze(
            1
        )  # we concat trough -1 axis, to expand embeding dim

        # (b×n×m)  and (b×m×p)
        # задача сделать выжимку из того что есть в енкодер аутпуте для каждого слова
        attn_applied = torch.bmm(
            attn_weights,  # bs x 1 x max_len
            encoder_outputs,  # bs x max_len x hidden_size
        ).squeeze(1)
        # attn_applied [bs x hidden_size]
        output = torch.cat((embedded_input, attn_applied), -1)
        output = self.attn_combine(output)

        # output.shape == [bs x hidden_size]
        # hidden.shape == [bs x hidden_size]
        output = F.relu(output)
        return output.unsqueeze(1), attn_weights

    def is_bidirect(self, hidden):
        if hidden.shape[0] == 2:
            return True
        elif hidden.shape[0] == 1:
            return False
        else:
            raise ValueError


class AttnDecoderRNN(nn.Module):
    def __init__(
        self,
        hidden_size,
        vocab_size,
        num_layers=1,
        dropout_p=0.1,
        max_length=MAX_LENGTH,
        bidirect=False,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        torch.manual_seed(0)
        self.dropout = nn.Dropout(dropout_p)
        torch.manual_seed(0)
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirect,
        )

        torch.manual_seed(0)
        self.attention = Attention(hidden_size, max_length, bidirect=bidirect)

        torch.manual_seed(0)
        bid_coef = 2 if bidirect else 1
        self.out = nn.Linear(hidden_size * bid_coef, vocab_size)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, input, hidden, encoder_outputs):
        """

        Parameters
        ----------
        input: tokens [bs, 1]
        hidden: hidden [bid_coef * layers_num, bs, hidden_size]
        encoder_outputs [bs, max_len, bid_coef * hidden_size]

        Returns
        -------

        """
        embedded = self.embedding(input).squeeze(1)
        embedded = self.dropout(embedded)

        # embedded_input, hidden, encoder_outputs
        attn_out, attn_weights = self.attention(embedded, hidden, encoder_outputs)

        output, hidden_out = self.gru(attn_out, hidden)

        output = F.log_softmax(
            self.out(output.squeeze(1)), dim=1
        )  # torch.Size([1, vocab_size])
        return output, hidden_out, attn_weights


def inference(batch, encoder, decoder, output_lang, max_length=MAX_LENGTH):
    with torch.no_grad():
        encoder_outputs, encoder_hidden = encoder(batch.enc)

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach().unsqueeze(0).unsqueeze(0)
    return decoded_words, decoder_attentions[: di + 1]
