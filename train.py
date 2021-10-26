import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from config import get_args
import models
from dataloader import DataLoader
import os
import pickle

class Classifier():
  def __init__(self, data, model_name, exp_name,test):

    self.data = data
    self.test = test
    self.exp_name = exp_name
    self.checkpoint_filepath = os.path.join("../checkpoints/", exp_name)
    self.model = getattr(models, model_name)
    self.model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'binary_crossentropy',
              metrics=['AUC', 'accuracy'])
    
  def train_model(self, epochs):
    print("="*80 + "Training model" + "="*80)
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=self.checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
    
    log_dir = os.path.join("../tensorboard/", self.exp_name)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    history = self.model.fit(self.data.train_generator,
        epochs=epochs,
        verbose=1,
        class_weight={0: 1., 1: 4.},
        validation_data = self.data.val_generator,
        callbacks=[model_checkpoint_callback, tensorboard_callback])
    
    self.eval_model()

  def binary_get_fp_and_fn_filenames(self, val_data):
      ground_truth = val_data.labels
      prob_predicted = self.model.predict(val_data)

      #in the multi-class case, we would use np.argmax below
      binary_predict = [0 if x[0] < 0.5 else 1 for x in prob_predicted]
      diff = ground_truth - binary_predict
      print('Confusion Matrix')
      print(confusion_matrix(val_data.classes, binary_predict))

      #Get the filepaths for false positives, false negatives, and false positives
      filepaths = np.array(val_data.filepaths) #Won't work if shuffle was set to true for val_data
      correct = filepaths[np.where(diff == 0)[0]]
      fp = filepaths[np.where(diff == -1)[0]]
      fn = filepaths[np.where(diff == 1)[0]]
      fp_and_fn_filenames = {
        'Correct': correct,
        'False positives': fp,
        'False negatives': fn
      }
      pickle.dump(fp_and_fn_filenames, open(os.path.join(self.checkpoint_filepath,
                                                      'filename_results.p'), 'wb'))

      print_all = False
      if print_all:
          print('Correctly classified: ', correct)
          print('False positives: ', fp)
          print('False negatives: ', fn)


  def eval_model(self):
    print("="*80 + "Evaluating model" + "="*80)
    if self.test:
      print("Restoring model weights from ", self.checkpoint_filepath)
      loaded_model = tf.keras.models.load_model(self.checkpoint_filepath)
      self.model = loaded_model
      self.model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'binary_crossentropy',
              metrics=['accuracy', 'AUC'])

    val_data = self.data.val_generator
    self.binary_get_fp_and_fn_filenames(val_data)
    self.model.evaluate(val_data)



def main(args):
  print("Num GPUs Available: ", tf.test.is_gpu_available())
  data = DataLoader()
  # data.fit()

  classifier = Classifier(data, model_name=args.model_name,
                          exp_name=args.exp_name, test=args.test)
  if args.test:
    classifier.eval_model()
  else:
    classifier.train_model(epochs=args.epochs)
  

if __name__ == '__main__':
    args = get_args()
    main(args)