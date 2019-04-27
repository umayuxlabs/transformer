from flask import Flask, request, jsonify
import logging
from flask_cors import CORS
import sys
import os

sys.path.append(os.path.abspath("/src/"))

logging.basicConfig(
    filename="/src/app/api.log",
    level=logging.DEBUG,
    format="%(levelname)s %(asctime)s - %(message)s",
)
log = logging.getLogger()

import tensorflow_datasets as tfds
import tensorflow as tf
import utensor.dataset as dt
from utensor.optimizer import CustomSchedule, loss_function
from utensor.model import Transformer
import time
from utensor.masking import create_masks
import pickle
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

checkpoint_path = "/src/data/banco/"
d_model = 128
MAX_LENGTH = 60
BUFFER_SIZE = 20000
BATCH_SIZE = 64
num_heads = 8
num_layers = 4
d_model = 128
dff = 512
dropout_rate = 0.1


def restore():

    # loading tokenizers for future predictions
    tokenizer_source = pickle.load(
        open(checkpoint_path + "./tokenizer_source.pickle", "rb")
    )
    tokenizer_target = pickle.load(
        open(checkpoint_path + "./tokenizer_target.pickle", "rb")
    )

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
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!!")
    else:
        print("Initializing from scratch.")

    return transformer, tokenizer_source, tokenizer_target


def evaluate(inp_sentence):
    start_token = [tokenizer_source.vocab_size]
    end_token = [tokenizer_source.vocab_size + 1]

    # inp sentence is portuguese, hence adding the start and end token
    inp_sentence = start_token + tokenizer_source.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

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


def translate(sentence):
    result, attention_weights = evaluate(sentence)

    predicted_sentence = tokenizer_target.decode(
        [i for i in result if i < tokenizer_target.vocab_size]
    )

    log.debug("Pregunta: {}".format(sentence))
    log.debug("Respuesta UmyBot: {}".format(predicted_sentence))

    return predicted_sentence


def rep_h(word):
    if "@" in word:
        return "[USER]"
    else:
        return word


def replace_identity(sentence):
    return " ".join([rep_h(i) for i in sentence.split()])


log.info("loading model ...")
transformer, tokenizer_source, tokenizer_target = restore()
log.info("model loaded")


@app.route("/", methods=["GET"])
def home():
    return """ 
        <h1>UmayuxLabs transformer</h1>
        <p>This is the implementation of the transformes (attention is all you need from google)
        made by UmayuxLabs.</p>
        <p>This API has an endpoint where you can retrieve a response to a text input</p>
        <p>Depending on the model you will receive a different output. This is a generic API</p>
        <h3>USAGE</h3>
        <p>
            POST: /predict/ <br>
                QUERY: {sentence: "..."} <br>
            RETURN: {response: "...", status: 1} <br><br>
        
        <strong>EXAMPLE</strong> <br>
        curl -X POST http://localhost:65431/predict/ -d'{"sentence": "hola, buenos dias"}' -H "Content-Type: application/json"
        <br><br>
        gustavo1$ curl -X POST https://www.umayuxlabs.com/api/v1/chatbot/assistant/predict/ -d'{"sentence": "hola, buenos dias"}' -H "Content-Type: application/json"
        </p>
    """


@app.route("/predict/", methods=["POST"])
def predict():
    data = request.get_json()
    log.debug(data)
    response = translate(data["sentence"])
    return jsonify({"response": replace_identity(response)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
