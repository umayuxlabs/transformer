import tensorflow_datasets as tfds
import tensorflow as tf
import src.dataset as dt
from src.optimizer import CustomSchedule, loss_function
from src.model import Transformer
import time
from src.masking import create_masks
import pickle
import matplotlib.pyplot as plt


class PredictModel(object):
    def __init__(
        self,
        MAX_LENGTH=40,
        BUFFER_SIZE=20000,
        BATCH_SIZE=64,
        EPOCHS=10,
        num_heads=8,
        num_layers=4,
        d_model=128,
        dff=512,
        dropout_rate=0.1,
        test_partition=0.2,
        dataset_file="./data/bancobot.tsv",
        checkpoint_path="./data/checkpoints/train/",
    ):

        self.MAX_LENGTH = MAX_LENGTH
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_model = d_model
        self.dff = dff
        self.dropout_rate = (dropout_rate,)
        self.test_partition = (test_partition,)
        self.dataset_file = dataset_file
        self.checkpoint_path = checkpoint_path

    def load(self):

        # loading tokenizers for future predictions
        with open(self.checkpoint_path + "/tokenizer_source.pickle", "rb") as handle:
            self.tokenizer_source = pickle.load(handle)

        with open(self.checkpoint_path + "/tokenizer_target.pickle", "rb") as handle:
            self.tokenizer_target = pickle.load(handle)

        self.learning_rate = CustomSchedule(self.d_model)
        self.optimizer = tf.keras.optimizers.Adam(
            self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
        )

        self.input_vocab_size = self.tokenizer_source.vocab_size + 2
        self.target_vocab_size = self.tokenizer_target.vocab_size + 2

        self.transformer = Transformer(
            self.num_layers,
            self.d_model,
            self.num_heads,
            self.dff,
            self.input_vocab_size,
            self.target_vocab_size,
            self.dropout_rate,
        )

        ckpt = tf.train.Checkpoint(
            transformer=self.transformer, optimizer=self.optimizer
        )
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, self.checkpoint_path, max_to_keep=5
        )

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Latest checkpoint restored!!")
        else:
            print("Initializing from scratch.")

    def evaluate(self, inp_sentence):
        start_token = [self.tokenizer_source.vocab_size]
        end_token = [self.tokenizer_source.vocab_size + 1]

        # inp sentence is portuguese, hence adding the start and end token
        inp_sentence = (
            start_token + self.tokenizer_source.encode(inp_sentence) + end_token
        )
        encoder_input = tf.expand_dims(inp_sentence, 0)

        # as the target is english, the first word to the transformer should be the
        # english start token.
        decoder_input = [self.tokenizer_target.vocab_size]
        output = tf.expand_dims(decoder_input, 0)

        for _ in range(self.MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output
            )

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = self.transformer(
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
        if tf.equal(predicted_id, self.tokenizer_target.vocab_size + 1):
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def plot_attention_weights(self, attention, sentence, result, layer):
        fig = plt.figure(figsize=(16, 8))

        sentence = self.tokenizer_source.encode(sentence)

        attention = tf.squeeze(attention[layer], axis=0)

        for head in range(attention.shape[0]):
            ax = fig.add_subplot(2, 4, head + 1)

            # plot the attention weights
            ax.matshow(attention[head][:-1, :], cmap="viridis")

            fontdict = {"fontsize": 10}

            ax.set_xticks(range(len(sentence) + 2))
            ax.set_yticks(range(len(result)))

            ax.set_ylim(len(result) - 1.5, -0.5)

            ax.set_xticklabels(
                ["<start>"]
                + [self.tokenizer_source.decode([i]) for i in sentence]
                + ["<end>"],
                fontdict=fontdict,
                rotation=90,
            )

            ax.set_yticklabels(
                [
                    self.tokenizer_target.decode([i])
                    for i in result
                    if i < self.tokenizer_target.vocab_size
                ],
                fontdict=fontdict,
            )

            ax.set_xlabel("Head {}".format(head + 1))

        plt.tight_layout()
        plt.show()

    def predict(self, sentence, plot=True):
        result, attention_weights = self.evaluate(sentence)

        predicted_sentence = self.tokenizer_target.decode(
            [i for i in result if i < self.tokenizer_target.vocab_size]
        )

        print("Pregunta: {}".format(sentence))
        print("Respuesta: {}".format(predicted_sentence))

        if plot:
            self.plot_attention_weights(attention_weights, sentence, result, plot)

