## Evaluation of trained models

Below are listed the results of running the ```test_eval.py``` script for each trained model against our own test set (20% of the initial data). 

All the models have been trained on our own training set (60% of the initial data). The models are trained for 25 epochs using a batch size of 64. For all models we use ```BCEWithLogitsLoss``` loss function, ```SGD``` optimizer and ```OneCycle``` scheduler (documented in detail above).

In the results we list also ```Ratio of F1``` for each model. We used ```Ratio of F1``` as a measure of overfitting and it was computed as a ratio of training and validation F1-scores after epoch 25. 

Results for models that utilize data augmentation techniques.

| Model | F1-score | Precision | Recall | Ratio of F1 | Training time |
| --- | --- | --- | --- | --- | --- | 
| Feed forward 1-layer | 0.245 | 0.191 | 0.341 | 0.881 | 22min 57s | 
| Feed forward 2-layer | 0.269 | 0.206 | 0.387 | 0.989 | 24min 9s | 
| Convolutional | 0.145 | 0.118 | 0.186 | 0.879 | 26min 55s | 
| Plain ResNet-152 | --- | --- | --- | --- | --- |

Results for models that utilize transfer learning and data augmentation techniques.

| Model | F1-score | Precision | Recall | Ratio of F1 | Training time | 
| --- | --- | --- | --- | --- | --- | 
| VGG16 | 0.700 | 0.626 | 0.794 | 1.014 | 27min 29s |   
| VGG16 with BN | 0.669 | 0.565 | 0.820 | 0.967 |  27min 44s | 
| ResNet-34 | 0.726 | 0.684 | 0.774 | 1.137 | 27min 36s | 
| ResNet-50 | 0.749 | 0.727 | 0.772 | 1.161 | 29min 29s | 
| ResNet-101 | 0.745 | 0.718 | 0.773 | 1.168 | 36min 57s | 
| ResNet-152 | 0.759 | 0.729 | 0.791 | 1.161 | 42min 22s | 

Results for models that utilize transfer learning (without data augmentation).

| Model | F1-score | Precision | Recall | Ratio of F1 | Training time |
| --- | --- | --- | --- | --- | --- | 
| ResNet-50 no aug | 0.751 | 0.783 | 0.722 | 1.350 | 14min 4s | 
| ResNet-50 no aug with dropout | --- | --- | --- | --- | --- | 

