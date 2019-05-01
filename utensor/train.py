import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=str, required=True)
    parser.add_argument("--buffer_size", type=int, required=True)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--learning_rate", default=1e-05, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--max_patience", default=5, type=int)

    return parser
