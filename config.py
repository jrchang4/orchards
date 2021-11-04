from argparse import ArgumentParser
import datetime

def get_args():
    psr = ArgumentParser()
    psr.add_argument("--epochs", type=int, default=8, help="Number of epochs to train model.")
    psr.add_argument("--model-name", type=str, default="fully_connected", choices = ["fully_connected", "basic_cnn", "Multimodal", "Inception"], help="Model name (in models.py)")
    psr.add_argument("--exp_name", type=str, default= datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), help="Name of the experiment")
    psr.add_argument("--checkpoint", type=str, default= datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), help="Path to checkpoint directory")
    
    psr.add_argument("--test", action='store_true')
    psr.add_argument("--batch-size", type=int, default=64, help="Number of batches.")
    psr.add_argument("--task", type=str, default="full", choices = ["full", "palm"], help="Full task Orchards vs Forests, or Palm vs Forests")
    psr.add_argument("--dropout", type=float, default=0.2, help="Dropout percentage.")
    psr.add_argument("--reg", type=float, default=0.1, help="Regularization constraint.")
    return psr.parse_args()
