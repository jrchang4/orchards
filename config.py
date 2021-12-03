from argparse import ArgumentParser
import datetime

def get_args():
    psr = ArgumentParser()
    psr.add_argument("--epochs", type=int, default=8, help="Number of epochs to train model.")
    psr.add_argument("--model-name", type=str, default="fully_connected", choices = ["fully_connected", "basic_cnn", "AlexNet", "Xception", "Inception", "ResNet", "vgg_conv", "Multimodal"], help="Model name (in models.py)")
    psr.add_argument("--exp-name", type=str, default= datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), help="Name of the experiment")
    psr.add_argument("--test", action='store_true')
    psr.add_argument("--reload", action='store_true')
    psr.add_argument("--batch-size", type=int, default=64, help="Number of batches.")
    psr.add_argument("--task", type=str, default="full", choices = ["full", "palm", "planet"], help="Full task Orchards vs Forests, or Palm vs Forests, or Planet full task")
    psr.add_argument("--dropout", type=float, default=0.2, help="Dropout percentage.")
    psr.add_argument("--reg", type=float, default=0.0, help="Regularization constraint.")
    psr.add_argument("--fine-tune", action="store_true", help="Fine Tune Model")
    return psr.parse_args()
