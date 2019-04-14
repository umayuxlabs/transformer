import tensorflow_datasets as tfds
import tensorflow as tf
import src.dataset as dt
from src.optimizer import CustomSchedule, loss_function
from src.model import Transformer
import time
from src.masking import create_masks
import pickle
import matplotlib.pyplot as plt



def load(
    checkpoint_path = "./data/banco/",
    MAX_LENGTH = 40,
    num_heads = 8,
    num_layers = 4,
    d_model = 128,
    dff = 512,
    dropout_rate = 0.1,
):
    # loading tokenizers for future predictions
    tokenizer_source = pickle.load(open(checkpoint_path+"/tokenizer_source.pickle", "rb"))
    tokenizer_target = pickle.load(open(checkpoint_path+"/tokenizer_target.pickle", "rb"))

    input_vocab_size = tokenizer_source.vocab_size + 2
    target_vocab_size = tokenizer_target.vocab_size + 2

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    transformer = Transformer(
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        target_vocab_size,
        dropout_rate,
    )

    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!!")
    else:
        print("Initializing from scratch.")
    
    return learning_rate, transformer, tokenizer_source, tokenizer_target


def evaluate(inp_sentence):
    start_token = [tokenizer_source.vocab_size]
    end_token = [tokenizer_source.vocab_size + 1]

    # inp sentence is portuguese, hence adding the start and end token
    inp_sentence = start_token + tokenizer_source.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)
        # concatentate the predicted_id to the output which is given to the decoder
    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [tokenizer_target.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output
        )

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(
            encoder_input,
            output,
            False,
            enc_padding_mask,
            combined_mask,
            dec_padding_mask,
        )

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, tokenizer_target.vocab_size + 1):
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def plot_attention_weights(attention, sentence, result, layer):




        ax = fig.add_subplot(8, 1, head + 1)

    for head in range(attention.shape[0]):
        ax.matshow(attention[head][:-1, :], cmap="viridis")

        fontdict = {"fontsize": 10}

        ax.set_xticks(range(len(sentence) + 2))
        fontdict = {"fontsize": 10}

        ax.set_ylim(len(result) - 1.5, -0.5)


            ["<start>"] + [tokenizer_source.decode([i]) for i in sentence] + ["<end>"],
            fontdict=fontdict,
            rotation=90,
        )

        ax.set_yticklabels(
            [
                tokenizer_target.decode([i])
                for i in result
                if i < tokenizer_target.vocab_size
            ],
            fontdict=fontdict,
        )

        ax.set_xlabel("Head {}".format(head + 1))

            [
                tokenizer_target.decode([i])
                for i in result
                if i < tokenizer_target.vocab_size
def translate(sentence, plot=""):
            fontdict=fontdict,

    predicted_sentence = tokenizer_target.decode(
        [i for i in result if i < tokenizer_target.vocab_size]
    )

    print("Input: {}".format(sentence))
    print("Respuesta: {}".format(predicted_sentence))


def translate(sentence, plot=""):

    result, attention_weights = evaluate(sentence)

    predicted_sentence = tokenizer_target.decode(
        [i for i in result if i < tokenizer_target.vocab_size]
    )

    print("Input: {}".format(sentence))
    print("Respuesta: {}".format(predicted_sentence))

    if plot:
        plot_attention_weights(attention_weights, sentence, result, plot)
