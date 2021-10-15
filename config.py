from argparse import ArgumentParser

def get_args():
    psr = ArgumentParser()
    psr.add_argument("--epochs", type=int, default=8, help="Number of epochs to train model.")
    psr.add_argument("--model-name", type=str, default="fully_connected", help="Model name (in models.py)")
    return psr.parse_args()