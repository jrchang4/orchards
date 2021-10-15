import tensorflow as tf
import numpy as np
from config import get_args
import models
from dataloader import DataLoader

class Classifier():
  def __init__(self, data, model_name):

    self.data = data
    self.model = getattr(models, model_name)
    self.model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'binary_crossentropy',
              metrics=['accuracy'])
    
  def train_model(self, epochs):
    print("="*80 + "Training model" + "="*80)
    
    history = self.model.fit(self.data.train_generator,
        steps_per_epoch=8,  
        epochs=epochs,
        verbose=1,
        validation_data = self.data.val_generator,
        validation_steps=8)
  
  def eval_model(self):
    print("="*80 + "Evaluating model" + "="*80)
    self.model.evaluate(self.data.val_generator)

def main(args):
  data = DataLoader()
  classifier = Classifier(data, model_name=args.model_name)
  classifier.train_model(epochs = args.epochs)
  classifier.eval_model()

if __name__ == '__main__':
    args = get_args()
    main(args)