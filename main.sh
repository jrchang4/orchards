#Run Xception Model
python train.py --model-name Xception --epochs $1 --fine-tune
#Run multimodal model
python train.py --model-name Multimodal --epochs $1 --fine-tune
