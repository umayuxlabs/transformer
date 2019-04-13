import tensorflow_datasets as tfds
import tensorflow as tf


examples, metadata = tfds.load(
    "ted_hrlr_translate/pt_to_en", with_info=True, as_supervised=True
)

train_examples, val_examples = examples["train"], examples["validation"]

tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13
)

tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13
)


def encode(lang1, lang2):
    """
        Add a start and end token to the input and target.
    """
    lang1 = (
        [tokenizer_pt.vocab_size]
        + tokenizer_pt.encode(lang1.numpy())
        + [tokenizer_pt.vocab_size + 1]
    )

    lang2 = (
        [tokenizer_en.vocab_size]
        + tokenizer_en.encode(lang2.numpy())
        + [tokenizer_en.vocab_size + 1]
    )

    return lang1, lang2


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)


def tf_encode(pt, en):
    return tf.py_function(encode, [pt, en], [tf.int64, tf.int64])


MAX_LENGTH = 40
BUFFER_SIZE = 20000
BATCH_SIZE = 64
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
# cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(
    BATCH_SIZE, padded_shapes=([-1], [-1])
)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(
    BATCH_SIZE, padded_shapes=([-1], [-1])
)


input_vocab_size = tokenizer_pt.vocab_size + 2
target_vocab_size = tokenizer_en.vocab_size + 2

