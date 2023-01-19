from seq2seq_attn.constants import device
from seq2seq_attn.data_prep import prepareData
from seq2seq_attn.eval import evaluateAndShowAttention, evaluateRandomly
from seq2seq_attn.models import AttnDecoderRNN, EncoderRNN
from seq2seq_attn.train import train

if __name__ == "__main__":
    input_lang, output_lang, pairs = prepareData('3', '3', False, filter=False)

    hidden_size = 8
    bidirect = True
    encoder = EncoderRNN(vocab_size=input_lang.n_words, hidden_size=hidden_size, num_layers=1, bidirect=bidirect).to(
        device)
    decoder = AttnDecoderRNN(hidden_size=hidden_size, vocab_size=output_lang.n_words, bidirect=bidirect).to(device)

    train(input_lang, output_lang, pairs, encoder, decoder, 1111, batch_size=22, n_epochs=1, learning_rate=.01)

    evaluateAndShowAttention(input_lang, output_lang, encoder, decoder, pairs[18], "epoch_!.png")

    evaluateRandomly(input_lang, output_lang, pairs, encoder, decoder, n=1)
