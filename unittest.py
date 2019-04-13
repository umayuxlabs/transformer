import logging
import sys

file_handler = logging.FileHandler(filename="tmp.log")
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [stdout_handler, file_handler]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=handlers,
)

logger = logging.getLogger()
logger.info("start")

import tensorflow_datasets as tfds
import tensorflow as tf
import time
import numpy as np

from src.positional_encoding import positional_encoding
from src.masking import create_padding_mask, create_look_ahead_mask
from src.attention import scaled_dot_product_attention, MultiHeadAttention
from src.layers import point_wise_feed_forward_network
from src.model import Transformer


def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(q, k, v, None)
    logger.info("Attention weights are:")
    logger.info(temp_attn)
    logger.info("Output is:")
    logger.info(temp_out)


logger.info(("using tensorflow: ", tf.__version__))

logger.info("Positional Encoding")
pos_encoding = positional_encoding(50, 512)
logger.info(pos_encoding.shape)
logger.info("\n\n\n\n")

logger.info("padding mask")
x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
logger.info(create_padding_mask(x))
logger.info("\n\n\n\n")

x = tf.random.uniform((1, 3))
logger.info("look aheadmask")
temp = create_look_ahead_mask(x.shape[1])
logger.info(temp)
logger.info("\n\n\n\n")

logger.info("scaled dot product attention")
np.set_printoptions(suppress=True)
temp_k = tf.constant(
    [[10, 0, 0], [0, 10, 0], [0, 0, 10], [0, 0, 10]], dtype=tf.float32
)  # (4, 3)
temp_v = tf.constant([[1, 0], [10, 0], [100, 5], [1000, 6]], dtype=tf.float32)  # (4, 3)
temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
print_out(temp_q, temp_k, temp_v)
logger.info("END\n\n\n\n")

logger.info("Multihead attention")
temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
out, attn = temp_mha(y, k=y, q=y, mask=None)
logger.info((out.shape, attn.shape))
logger.info("END\n\n\n\n")

logger.info("Point wise forward network")
sample_ffn = point_wise_feed_forward_network(512, 2048)
logger.info(sample_ffn(tf.random.uniform((64, 50, 512))).shape)
logger.info("END\n\n\n\n")


# logger.info("Test Transformer")
# sample_transformer = Transformer(
#     num_layers=2,
#     d_model=64,
#     num_heads=1,
#     dff=2048,
#     input_vocab_size=8500,
#     target_vocab_size=8000,
# )
# temp_input = tf.random.uniform((1, 62))
# temp_target = tf.random.uniform((1, 26))
# fn_out, fn_att = sample_transformer(
#     temp_input,
#     temp_target,
#     training=False,
#     enc_padding_mask=None,
#     look_ahead_mask=None,
#     dec_padding_mask=None,
# )

# logger.info(
#     "output weights: {}, attention_weights: {}".format(fn_out.shape)
# )  # (batch_size, tar_seq_len, target_vocab_size)
# logger.info("END\n\n\n\n")


import src.dataset as dt

logger.info("Dataset Testing")
BUFFER_SIZE = 20000
BATCH_SIZE = 32

dataset = dt.Dataset(filename="./data/test.tsv")
dataset.build_train_test(test=0.2)
train_examples, test_examples = dataset.format_train_test()
tokenizer_source, tokenizer_target = dataset.tokenizer(train_examples)

train_dataset = train_examples.map(dataset.tf_encode)

test_dataset = test_examples.map(dataset.tf_encode)


logger.info("######## Source")

sample_string = dt.preprocess_sentence(
    [i.numpy().decode("UTF-8").split("\t")[0].encode() for i in train_examples][0]
)

tokenized_string = tokenizer_source.encode(sample_string)
logger.info("Tokenized string is {}".format(tokenized_string))

original_string = tokenizer_source.decode(tokenized_string)
logger.info("The original string: {}".format(original_string))

for ts in tokenized_string:
    logger.info("{} ----> {}".format(ts, tokenizer_source.decode([ts])))

logger.info([i for i in train_dataset][0])

logger.info("######## Target")
sample_string = dt.preprocess_sentence(
    [i.numpy().decode("UTF-8").split("\t")[1].encode() for i in train_examples][0]
)

tokenized_string = tokenizer_target.encode(sample_string)
logger.info("Tokenized string is {}".format(tokenized_string))

original_string = tokenizer_target.decode(tokenized_string)
logger.info("The original string: {}".format(original_string))

for ts in tokenized_string:
    logger.info("{} ----> {}".format(ts, tokenizer_target.decode([ts])))

logger.info([i for i in train_dataset][0][1])

logger.info("END\n\n\n\n")
