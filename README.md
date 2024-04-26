# ContextualStructuralSimilarity

Official code for our Pattern Recognition paper Improving image segmentation with contextual and structural similarity.

Link to paper: https://www.sciencedirect.com/science/article/pii/S0031320324002401?dgcid=rss_sd_all

In the released code, we demonstrate the use of our method for pancreas segmentation using NIH pancreas dataset. However, the same method can be used for other organs accurately and efficiently. To use the code, follow the 3 steps below.

Step1: Data preprocessing 

       Don't worry about it! Only spatial normalization and intensity clipping are needed.
       
       Images are named as "image0001.nii.gz", while labels are named as "label0001.nii.gz".
       
       After processing, save them where you can access and change the paths in training.py and testing.py

Step2: Training

       Command: python training

Step3: Inference

       Command: python testing


A pretrained a model is previded in checkpoints folder. It is trained with images from 21 to 82. When using it for inference, you can expect an average DSC of 85.4% on the first fold.
