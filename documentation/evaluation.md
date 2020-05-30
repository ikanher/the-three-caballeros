| Model | F1-score | Precision | Recall | Time
| --- | --- | --- | --- | --- | 
| Feed forward 1-layer | 0.245 | 0.191 | 0.341 | 22min 57s | 
| Feed forward 2-layer | 0.269 | 0.206 | 0.387 | 24min 9s | 
| Convolutional | 0.145 | 0.118 | 0.186 | 26min 55s | 
| Plain ResNet-152 | --- | --- | --- | --- | 

Results for models that use transfer learning and data augmentation.

| Model | F1-score | Precision | Recall |--- | 
| --- | --- | --- | --- | --- | 
| VGG16 | 0.700 | 0.626 | 0.794 |--- |   
| VGG16 with BN | 0.669 | 0.565 | 0.820 | 27min 44s | 
| ResNet-18 | --- | --- | --- | --- | 
| ResNet-34 | --- | --- | --- | --- | 
| ResNet-50 | 0.749 | 0.727 | 0.772 | 29min 29s | 
| ResNet-101 | 0.745 | 0.718 | 0.773 | 36min 57s | 
| ResNet-152 | --- | --- | --- | --- | 



