from argparse import ArgumentParser
import datetime

def get_args():
    psr = ArgumentParser()
    psr.add_argument("--epochs", type=int, default=8, help="Number of epochs to train model.")
    psr.add_argument("--model-name", type=str, default="fully_connected", choices = ["fully_connected", "basic_cnn"], help="Model name (in models.py)")
    psr.add_argument("--exp_name", type=str, default= datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), help="Name of the experiment")
    psr.add_argument("--test", action='store_true')
    return psr.parse_args()