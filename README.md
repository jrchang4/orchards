# orchards

To run a model:

python train.py

Use --epochs flag to specify the number of epochs.
Use --model-name flag to specify the model in models.py to train.
Use --checkpoint flag to specify the name of the saved model checkpoint.

To evaluate a model:

python train --checkpoint {checkpoint} --test

Use the --test flag to evaluate a model
Must include a checkpoint to load the model to test (just checkpoint name, not entire checkpoint)

Example Flow:

python train.py --model-name basic_cnn --checkpoint trial1
python train.py --checkpoint trial1 --test

Tensorboard:

Local - From the parent directory of the 'orchards' git repository, run:

tensorboard --logdir tensorboard

Open the provided link in a web browser (http://localhost:6006/)

Remote VM

Mount data in VM:

gcloud auth application-default login
gcsfuse --implicit-dirs es262-orchards-forests data
