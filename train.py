import tensorflow_datasets as tfds
import tensorflow as tf

from src.optimizer import CustomSchedule, loss_function
from src.dataset import Dataset
from src.model import Transformer
import time
from src.masking import create_masks
import pickle


# define training function step
@tf.function
def train_step(inp, tar, transformer, optimizer, train_loss, train_accuracy):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(
            inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask
        )
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


class TrainModel(object):
    def train(
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

        # Build the dataset for training validation
        dataset = Dataset(filename=dataset_file)
        dataset.build_train_test(test=test_partition)
        train_examples, val_examples = dataset.format_train_test()
        tokenizer_source, tokenizer_target = dataset.tokenizer(train_examples)

        # saving tokenizers
        with open(checkpoint_path + "/tokenizer_source.pickle", "wb") as handle:
            pickle.dump(tokenizer_source, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(checkpoint_path + "/tokenizer_target.pickle", "wb") as handle:
            pickle.dump(tokenizer_target, handle, protocol=pickle.HIGHEST_PROTOCOL)

        train_dataset = train_examples.map(dataset.tf_encode)
        train_dataset = train_dataset.filter(dataset.filter_max_length)
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(
            BATCH_SIZE, padded_shapes=([-1], [-1])
        )
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        val_dataset = val_examples.map(dataset.tf_encode)
        val_dataset = val_dataset.filter(dataset.filter_max_length).padded_batch(
            BATCH_SIZE, padded_shapes=([-1], [-1])
        )

        input_vocab_size = tokenizer_source.vocab_size + 2
        target_vocab_size = tokenizer_target.vocab_size + 2

        # Setup the learning rate and optimizer
        learning_rate = CustomSchedule(d_model)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
        )

        train_loss = tf.keras.metrics.Mean(name="train_loss")
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_accuracy"
        )

        # setup Transformer
        transformer = Transformer(
            num_layers,
            d_model,
            num_heads,
            dff,
            input_vocab_size,
            target_vocab_size,
            dropout_rate,
        )

        # setup checkpoints
        ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Latest checkpoint restored!!")
        else:
            print("Initializing from scratch.")

        # training loop
        for epoch in range(EPOCHS):
            start = time.time()

            train_loss.reset_states()
            train_accuracy.reset_states()

            # inp -> portuguese, tar -> english
            for (batch, (inp, tar)) in enumerate(train_dataset):
                train_step(inp, tar, transformer, optimizer, train_loss, train_accuracy)
                if batch % 500 == 0:
                    print(
                        "Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}".format(
                            epoch + 1,
                            batch,
                            train_loss.result(),
                            train_accuracy.result(),
                        )
                    )

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                print(
                    "Saving checkpoint for epoch {} at {}".format(
                        epoch + 1, ckpt_save_path
                    )
                )

            print(
                "Epoch {} Loss {:.4f} Accuracy {:.4f}".format(
                    epoch + 1, train_loss.result(), train_accuracy.result()
                )
            )

            print("Time taken for 1 epoch: {} secs\n".format(time.time() - start))
