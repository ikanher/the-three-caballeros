## Evaluation of trained models

All the models have been trained on our own training set (60% of all data) with batch size of 64 for 25 epochs. For all models we use ```BCEWithLogitsLoss``` loss function, ```SGD``` optimizer and ```OneCycle``` scheduler with the same hyperparameters (documented above).

Results for models that utilize data augmentation techniques.

| Model | F1-score | Precision | Recall | Training time |
| --- | --- | --- | --- | --- | 
| Feed forward 1-layer | 0.245 | 0.191 | 0.341 | 22min 57s | 
| Feed forward 2-layer | 0.269 | 0.206 | 0.387 | 24min 9s | 
| Convolutional | 0.145 | 0.118 | 0.186 | 26min 55s | 
| Plain ResNet-152 | --- | --- | --- | --- | 

Results for models that utilize transfer learning and data augmentation techniques.

| Model | F1-score | Precision | Recall | Training time | 
| --- | --- | --- | --- | --- | 
| VGG16 | 0.700 | 0.626 | 0.794 | 27min 29s |   
| VGG16 with BN | 0.669 | 0.565 | 0.820 | 27min 44s | 
| ResNet-34 | 0.726 | 0.684 | 0.774 | 27min 36s | 
| ResNet-50 | 0.749 | 0.727 | 0.772 | 29min 29s | 
| ResNet-101 | 0.745 | 0.718 | 0.773 | 36min 57s | 
| ResNet-152 | --- | --- | --- | --- | 

Results for models that utilize transfer learning (without data augmentation).

| Model | F1-score | Precision | Recall | Training time |
| --- | --- | --- | --- | --- | 
| ResNet-152 without augmentation | --- | --- | --- | --- | 

