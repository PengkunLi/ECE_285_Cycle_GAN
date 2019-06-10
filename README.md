# ECE_285_Cycle_GAN
================

This is project of Style Transfer with Neural Style Transfer and Cycle-GAN developed by Jixuan Liu, Yuhao Tian, Tuoyi Zhao and Yufei Li.

## Requirements:
================

Install package 'easydict' as follow:
        $ pip install --user easydict

## Code organization:
================

demo.ipynb          -- Run a demo which import the trained model from the checkpoint and then apply it on                          the test set and plot the first 8 images

NewCycleGAN.ipynb   -- Run the training, validation and test of our model

NeuralTransfer.ipynb   -- Use Neural Style Transfer model to achieve Style Transfer between two images

models.py           -- Basic architectures of Cycle-GAN (Generators and Discriminators) as mentioned in the 3.3.2 (Implementation) of the report.

utils.py            -- Include auxiliary functions which are needed during the training process of Cycle-GAN.

check0              -- Include some files which are needed when implement demo. 

Also, you need to download the checkpoint.pth.tar file into the check0 file if you want to implement demo.ipynb. The download dir is :https://drive.google.com/open?id=18M5HZIJX1iyvt95PmyJoXXyOr3r1AER0
