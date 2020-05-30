# TODO

## Basic setup
- [x] Dataloading
- [x] Training loop
- [x] Batch visualization
- [x] Accuracy calculation
- [x] Basic feedforward network
- [x] Basic CNN
- [x] Train/valid/test -split (Eeva-Maria)
- [x] What to do with BW images - convert to RGB and include them
- [x] Input data normalization (mean=0, std=1) (Eeva-Maria)
- [x] BatchNorm for FF NN (implemented in 01 and 04 -notebooks)
- [x] Report validation loss when training
- [x] Confusion matrix (Eeva-Maria)
- [x] Confusion matrix visulizations (Eeva-Maria)

## Some problems
- [x] Support for multi-label inputs
- [x] Multi-label loss

## Possible problems
- [x] When leaving out some data the normalization statistics need to be recalculated? (probably just going to use all the data now that we have pos weights)
- [x] What to do with imbalance of classes (`pos_weight` seems to work well)

## Improvements
- [x] Train, valid and test sets to have separate transformations
- [x] Data augmentation
- [x] Adjust BCEWithLogitsLoss with parameter `pos_weight` to compensate imbalanced labels
- [x] Improved CNN structure - probably better just to skip to pretrained models, training from scratch is not feasible
- [x] [One Cycle Policy](https://arxiv.org/pdf/1803.09820.pdf) (implemented in 08 -notebook)

## Transfer learning
- [x] VGG16 (currently approx 0.67 validation f1)
- [x] ResNet 18/34/50 - DOES NOT SEEM TO FIT
- [x] ResNet 101 - Finally starting to overfit with this, Train f1: 0.9421016, Valid f1: 0.7125345
- [x] ResNet 152

## Outscoped approaches
- Inception?
- Other models? [Torcvision models](https://pytorch.org/docs/stable/torchvision/models.html)
- More recent model, such as ResNet Inception v2 [not yet in torchvision](https://github.com/Cadene/pretrained-models.pytorch)?
- Ensembles, such as per-label majority-voting between 3 of our best models

## Final tweaks
- [x] Train with valid/test data

## Report
- [x] Explain what approaches and parameters you have tried
- [x] What worked well and what did not
- [x] *Error analysis*: present some representative examples, with possible explanations about what does not work and how it could be made to work better

## Required for the course staff:
- [x] Output for `test_eval.py` -script

## Submission
- [x] Project code
- [x] System output on test data
- [x] A short report / documentation
- [x] Return zip to roman.yangarber@helsinki.fi, lidia.pivovarova@helsinki.fi, anh-duc.vu@helsinki.fi
