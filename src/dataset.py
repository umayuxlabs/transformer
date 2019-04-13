import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import unicodedata
import re
import random

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def preprocess_sentence(w):
    return w
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    # w = '<start> ' + w + ' <end>'
    return w


class Dataset:
    def __init__(self, filename="", vocab_dim=10000, max_length=40, buffer_size=20000):
        self.input_filename = filename
        self.train_filename = filename + ".train"
        self.test_filename = filename + ".test"
        self.vocabulary_size = vocab_dim
        self.max_length = max_length
        self.buffer_size = buffer_size

    def build_train_test(self, test=0.2):
        """ This function takes the input file and splits into training and testing """
        file_lines = []
        for ix, i in enumerate(open(self.input_filename)):
            file_lines.append(ix)

        random.shuffle(file_lines)
        test_limit = int(len(file_lines) * test)

        test_dict = {i: True for i in file_lines[0:test_limit]}

        fo_test = open(self.test_filename, "w")
        fo_train = open(self.train_filename, "w")
        for ix, i in enumerate(open(self.input_filename)):
            try:
                assert test_dict[ix]
                fo_test.write(i)
            except Exception as e:
                fo_train.write(i)

    def tokenizer(self):
        filenames = [self.train_filename]
        dataset_tf = tf.data.Dataset.from_tensor_slices(filenames)
        self.train_dataset = dataset_tf.flat_map(
            lambda filename: (tf.data.TextLineDataset(filename))
        )

        self.tokenizer_source = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (
                preprocess_sentence(pt.numpy().decode("UTF-8").split("\t")[0]).encode()
                for pt in self.train_dataset
            ),
            target_vocab_size=self.vocabulary_size,
        )

        self.tokenizer_target = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (
                preprocess_sentence(pt.numpy().decode("UTF-8").split("\t")[1]).encode()
                for pt in self.train_dataset
            ),
            target_vocab_size=self.vocabulary_size,
        )

    def encode(self, string):
        source = (
            [self.tokenizer_source.vocab_size]
            + self.tokenizer_source.encode(
                preprocess_sentence(
                    string.numpy().decode("UTF-8").split("\t")[0]
                ).encode()
            )
            + [self.tokenizer_source.vocab_size + 1]
        )

        target = (
            [self.tokenizer_target.vocab_size]
            + self.tokenizer_target.encode(
                preprocess_sentence(
                    string.numpy().decode("UTF-8").split("\t")[1]
                ).encode()
            )
            + [self.tokenizer_target.vocab_size + 1]
        )

        return source, target

    def filter_max_length(self, x, y):
        """ # drop examples with more than max_length tokens """
        return tf.logical_and(
            tf.size(x) <= self.max_length, tf.size(y) <= self.max_length
        )

    def tf_encode(self, string):
        """
        Operations inside .map() run in graph mode and receive a graph tensor that do
        not have a numpy attribute. The tokenizer expects a string or Unicode symbol
        to encode it into integers. Hence, you need to run the encoding inside a
        tf.py_function, which receives an eager tensor having a numpy attribute that
        contains the string value.
        """
        return tf.py_function(self.encode, [string], [tf.int64, tf.int64])

    def get_train_dataset(self):
        self.train_tensor_dataset = self.train_dataset.map(self.tf_encode)

        self.train_tensor_dataset = self.train_tensor_dataset.filter(
            self.filter_max_length
        )

        # cache the dataset to memory to get a speedup while reading from it.
        self.train_tensor_dataset = self.train_tensor_dataset.cache()
        self.train_tensor_dataset = self.train_tensor_dataset.shuffle(
            self.buffer_size
        ).padded_batch(self.buffer_size, padded_shapes=([-1], [-1]))
        train_dataset = self.train_tensor_dataset.prefetch(
            tf.data.experimental.AUTOTUNE
        )

    def_get_val_dataset(self):

dataset = Dataset(filename="../data/test.tsv")
dataset.build_train_test(test=0.2)
dataset.tokenizer()
dataset.get_train_dataset()


sample_string = preprocess_sentence(
    b"sigue fallando el bot\xc3\xb3n de pagos #pse de @falabellayudaco @banco_falabella"
)

tokenized_string = dataset.tokenizer_source.encode(sample_string)
print("Tokenized string is {}".format(tokenized_string))

original_string = dataset.tokenizer_source.decode(tokenized_string)
print("The original string: {}".format(original_string))

for ts in tokenized_string:
    print("{} ----> {}".format(ts, dataset.tokenizer_source.decode([ts])))

