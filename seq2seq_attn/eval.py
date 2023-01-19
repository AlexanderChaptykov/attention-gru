import random

from matplotlib import pyplot as plt, ticker

from seq2seq_attn.data_prep import MAX_LENGTH
from seq2seq_attn.dataset import tensorsFromPair
from seq2seq_attn.models import EncoderRNN, AttnDecoderRNN, inference
from seq2seq_attn.train import collate_fn


def evaluate(
    input_lang,
    output_lang,
    encoder: EncoderRNN,
    decoder: AttnDecoderRNN,
    pair,
    max_length=MAX_LENGTH,
):
    two_lang = tensorsFromPair(input_lang, output_lang, pair)
    batch = collate_fn([two_lang])
    decoded_words, decoder_attentions = inference(batch, encoder, decoder, output_lang)
    return decoded_words, decoder_attentions


def evaluateRandomly(input_lang, output_lang, pairs, encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print(">", pair[0])
        print("=", pair[1])
        output_words, attentions = evaluate(
            input_lang, output_lang, encoder, decoder, pair
        )
        output_sentence = " ".join(output_words)
        print("<", output_sentence)
        print("")


def showAttention(input_sentence, output_words, attentions, filename="mygraph.png"):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap="bone")
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([""] + input_sentence.split(" ") + ["<EOS>"], rotation=90)
    ax.set_yticklabels([""] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.savefig(filename)


def evaluateAndShowAttention(
    input_lang, output_lang, encoder, decoder, pair, file_name
):
    output_words, attentions = evaluate(input_lang, output_lang, encoder, decoder, pair)
    input_sentence = pair[0]
    print("input =", input_sentence)
    print("output =", " ".join(output_words))

    showAttention(input_sentence, output_words, attentions, file_name)
    return output_words
