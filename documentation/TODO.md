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
- [ ] Other pre-processing (whitening, ...)?
- [x] BatchNorm for FF NN (implemented in 01 and 04 -notebooks)
- [x] Report validation loss when training
- [x] Confusion matrix (Eeva-Maria)
- [x] Confusion matrix visulizations (Eeva-Maria)

## Some problems
- [x] Support for multi-label inputs
- [x] Multi-label loss

## Possible problems
- [ ] When leaving out some data the normalization statistics need to be recalculated? (probably just going to use all the data now that we have pos weights)
- [x] What to do with imbalance of classes (pos\_weight seems to work well)

## Improvements
- [ ] Train, valid and test sets to have separate transformations
- [ ] Data augmentation
- [x] Adjust BCEWithLogitsLoss with parameter pow\_weight to compensate imbalanced labels
- [ ] Improved CNN structure - probably better just to skip to pretrained models, training from scratch is not feasible
- [x] [One Cycle Policy](https://arxiv.org/pdf/1803.09820.pdf) (implemented in 08 -notebook)

## Transfer learning
- [x] VGG16 (currently approx 0.67 validation f1)
- [x] ResNet 18/34/50 - DOES NOT SEEM TO FIT
- [x] ResNet 101 - Finally starting to overfit with this, Train f1: 0.9421016, Valid f1: 0.7125345
- [ ] ResNet 152 - Could be worth testing?
- [ ] Inception?
- [ ] Other models? [Torcvision models](https://pytorch.org/docs/stable/torchvision/models.html)
- [ ] More recent model, such as ResNet Inception v2 [not yet in torchvision](https://github.com/Cadene/pretrained-models.pytorch)?

## Required for the course staff:
- [ ] Report
- [ ] Output for `test_eval.py` -script
- [ ] Ground truth for `test_eval.py` -script
