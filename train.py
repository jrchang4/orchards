import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
import numpy as np
from config import get_args
import models
from dataloader import DataLoader
import os
from iterator import Iterator
import pickle


# def combine_gen(*gens):
#     while True:
#         for g in gens:
#             yield next(g)

class MergedGenerators(Sequence):
    def __init__(self, *generators):
        self.generators = generators
        # TODO add a check to verify that all generators have the same length

    def __len__(self):
        return len(self.generators[0])

    def __getitem__(self, index):
        return [generator[index] for generator in self.generators]

class Classifier():
  def __init__(self, data, model_name, exp_name,test):

    self.data = data
    self.test = test
    self.exp_name = exp_name
    self.checkpoint_filepath = os.path.join("../checkpoints/", exp_name)
    self.model = getattr(models, model_name)
    self.model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'binary_crossentropy',
              metrics=['AUC', 'accuracy', self.recall_m, self.precision_m, self.f1_m])

    


  def recall_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

  def precision_m(self, y_true, y_pred):
      true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
      predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
      precision = true_positives / (predicted_positives + K.epsilon())
      return precision

  def f1_m(self, y_true, y_pred):
      precision = self.precision_m(y_true, y_pred)
      recall = self.recall_m(y_true, y_pred)
      return 2*((precision*recall)/(precision+recall+K.epsilon()))


  def binary_get_fp_and_fn_filenames(self, val_data1, val_data2):
    c_matrix1, fp_and_fn_filenames1 = self.binary_get_fp_and_fn_filenames(val_data1)
    c_matrix2, fp_and_fn_filenames2 = self.binary_get_fp_and_fn_filenames(val_data2)

    combined = {}
    for type in fp_and_fn_filenames1.keys():
        combined[type] = fp_and_fn_filenames1[type]
        combined[type].extend(fp_and_fn_filenames2[type])

    pickle.dump(combined, open(os.path.join(self.checkpoint_filepath, 'filename_results.p'), 'wb'))

  def binary_get_fp_and_fn_filenames(self, val_data):
      ground_truth = val_data.labels
      prob_predicted = self.model.predict(val_data)

      #in the multi-class case, we would use np.argmax below
      binary_predict = [0 if x[0] < 0.5 else 1 for x in prob_predicted]
      diff = ground_truth - binary_predict
      print('Confusion Matrix')
      c_matrix = confusion_matrix(val_data.classes, binary_predict)
      print(c_matrix)

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

      return c_matrix, fp_and_fn_filenames
      # pickle.dump(fp_and_fn_filenames, open(os.path.join(self.checkpoint_filepath,
          #                                            'filename_results.p'), 'wb'))

      # print_all = False
      # if print_all:
      #     print('Correctly classified: ', correct)
      #     print('False positives: ', fp)
      #     print('False negatives: ', fn)
          
  def train_model(self, epochs, task):
    print("="*80 + "Training model" + "="*80)
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=self.checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
    
    log_dir = os.path.join("../tensorboard/", self.exp_name)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    history = self.model.fit(MergedGenerators(self.data.train_generator1, self.data.train_generator2),
        steps_per_epoch=len(self.data.train_generator1) + len(self.data.train_generator2),
        epochs=epochs,
        verbose=1,
        validation_data = MergedGenerators(self.data.val_generator1, self.data.val_generator2),
        callbacks=[model_checkpoint_callback, tensorboard_callback])
    
    self.eval_model()

  def eval_model(self):
    print("="*80 + "Evaluating model" + "="*80)
        
    if self.test:
      print("Restoring model weights from ", self.checkpoint_filepath)
      loaded_model = tf.keras.models.load_model(self.checkpoint_filepath, custom_objects={'recall_m':self.recall_m, 'precision_m':self.precision_m, 'f1_m':self.f1_m})
      self.model = loaded_model
      self.model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'binary_crossentropy',
              metrics=['accuracy', 'AUC', self.recall_m, self.precision_m, self.f1_m])

    val_data = MergedGenerators(self.data.val_generator1, self.data.val_generator2)
    self.binary_get_fp_and_fn_filenames(val_data)
    self.model.evaluate(val_data)

def main(args):
  print("Num GPUs Available: ", tf.test.is_gpu_available())

  data = DataLoader(batch_size = args.batch_size, task = args.task)
  classifier = Classifier(data, model_name=args.model_name,
                          exp_name=args.exp_name, test=args.test)
  if args.test:
    classifier.eval_model()
  else:
    classifier.train_model(epochs=args.epochs, task=args.task)
  

if __name__ == '__main__':
    args = get_args()
    main(args)
