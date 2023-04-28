# ContextualStructuralSimilarity

Step1: Data preprocessing 

       Don't worry about it! Only spatial normalization and intensity clipping are needed.
       
       Images are named as "image0001.nii.gz", while labels are named as "label0001.nii.gz".
       
       After processing, save them where you can access and change the paths in training.py and testing.py

Step2: Training

       Command: python training

Step3: Inference

       Command: python testing


A pretrained a model is previded in checkpoints folder. It is trained with images from 21 to 82. When using it for inference, you can expect an average DSC of 85.4% on the first folder
