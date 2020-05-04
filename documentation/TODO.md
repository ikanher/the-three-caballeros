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

## Improvements
- [ ] Adjust BCEWithLogitsLoss with parameter pow_weight to compensate imbalanced labels
- [ ] Use a vector of thresholds (different for different classes) for evaluation
- [ ] Improved CNN structure

## Future
- [ ] Data augmentation
- [ ] What to do with imbalance of classes (already listed pow_weight and thresholds; these should go quite far)
- [ ] Transfer learning (probably small RESNET to start with?)

## Required for the course staff:
- [ ] Report
- [ ] Output for `test_eval.py` -script
- [ ] Ground truth for `test_eval.py` -script
