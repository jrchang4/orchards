import tensorflow as tf
import numpy as np
from config import get_args
import models
from dataloader import DataLoader
import os

class Classifier():
  def __init__(self, data, model_name, checkpoint,test):

    self.data = data
    self.test = test
    self.checkpoint_filepath = os.path.join("../checkpoints/", checkpoint)
    self.model = getattr(models, model_name)
    self.model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'binary_crossentropy',
              metrics=['accuracy'])
    
  def train_model(self, epochs):
    print("="*80 + "Training model" + "="*80)
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=self.checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
    
    history = self.model.fit(self.data.train_generator,
        steps_per_epoch=8,  
        epochs=epochs,
        verbose=1,
        validation_data = self.data.val_generator,
        validation_steps=8,
        callbacks=[model_checkpoint_callback])
    
    self.eval_model()
  
  def eval_model(self):
    print("="*80 + "Evaluating model" + "="*80)
    if self.test:
      print("Restoring model weights from ", self.checkpoint_filepath)
      loaded_model = tf.keras.models.load_model(self.checkpoint_filepath)
      self.model = loaded_model
      self.model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'binary_crossentropy',
              metrics=['accuracy'])
    self.model.evaluate(self.data.val_generator)

def main(args):
  data = DataLoader()
  classifier = Classifier(data, model_name=args.model_name,
                          checkpoint=args.checkpoint, test=args.test)
  if args.test:
    classifier.eval_model()
  else:
    classifier.train_model(epochs=args.epochs)
  

if __name__ == '__main__':
    args = get_args()
    main(args)