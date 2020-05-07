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
- [ ] When leaving out some data the normalization statistics need to be recalculated?
- [x] What to do with imbalance of classes (pos\_weight seems to work well)

## Improvements
- [ ] Data augmentation
- [x] Adjust BCEWithLogitsLoss with parameter pow\_weight to compensate imbalanced labels
- [ ] Improved CNN structure - probably better just to skip to pretrained models, training from scratch is not feasible
- [ ] Learning rate finder/Cyclical learning rate - [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/pdf/1506.01186.pdf)

## Transfer learning
- [x] VGG16 (currently approx 0.67 validation f1)
- [ ] RESNET 18/34/50 (no good results yet)
- [ ] Inception?
- [ ] Other models? [Torcvision models](https://pytorch.org/docs/stable/torchvision/models.html)

## Required for the course staff:
- [ ] Report
- [ ] Output for `test_eval.py` -script
- [ ] Ground truth for `test_eval.py` -script
