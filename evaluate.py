import argparse
import tensorflow_datasets as tfds
import tensorflow as tf

from utensor.optimizer import CustomSchedule, loss_function
from utensor.dataset import Dataset
from utensor.model import Transformer
from utensor.dataset import load_dataset
from utensor.masking import create_masks
import pickle
from sklearn.metrics import classification_report
import time
import os
import json

tf.keras.backend.clear_session()


def restore(args):
    # loading tokenizers for future predictions
    tokenizer_source = pickle.load(
        open(args.checkpoint_path + "tokenizer_source.pickle", "rb")
    )
    tokenizer_target = pickle.load(
        open(args.checkpoint_path + "tokenizer_target.pickle", "rb")
    )

    input_vocab_size = tokenizer_source.vocab_size + 2
    target_vocab_size = tokenizer_target.vocab_size + 2

    learning_rate = CustomSchedule(args.d_model)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    transformer = Transformer(
        args.num_layers,
        args.d_model,
        args.num_heads,
        args.dff,
        input_vocab_size,
        target_vocab_size,
        args.dropout_rate,
    )

    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.checkpoint_path, max_to_keep=1)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!!")
    else:
        print("Initializing from scratch.")

    return transformer, tokenizer_source, tokenizer_target


def evaluate(inp_sentence, args):
    start_token = [tokenizer_source.vocab_size]
    end_token = [tokenizer_source.vocab_size + 1]

    # inp sentence is portuguese, hence adding the start and end token
    inp_sentence = start_token + tokenizer_source.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [tokenizer_target.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(args.MAX_LENGTH):
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


def translate(sentence, args):
    result, attention_weights = evaluate(sentence, args)

    predicted_sentence = tokenizer_target.decode(
        [i for i in result if i < tokenizer_target.vocab_size]
    )

    print("Pregunta: {}".format(sentence))
    print("Respuesta UmyBot: {}".format(predicted_sentence))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    params = json.load(open(args.checkpoint_path + "/params.json"))
    transformer, tokenizer_source, tokenizer_target = restore(args)
    translate("me pueden ayudar?", args)

