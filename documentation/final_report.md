# Final report
Project for course Deep Learning
University of Helsinki
Spring 2020

**Group name**: The Three Caballeros
**Group members**: Mikko Kotola, Eeva-Maria Laiho, Aki Rehn
**Chosen dataset**: Images

*To be removed: mandatory content of the report (from instructions)*
* Approaches and parameters the group tried
* What worked well and what did not
* Error analysis: present some representative examples, with possible explanations about what does not work and how it could be made to work better

## Short description of the final model and training process
*This will likely be a transfer learning model.*

### Preprocessing
* Read in images and labels so that we could attach a one-hot encoded vector of labels to each image
* Used images with no labels (n=9824 out of 20000)
* Normalized images in all three dataset using the means and stds of RGB channels of the whole image set
* NOT CURRENTLY RESIZING: (Resized images to 224x224 to enable using transfer learning models)
* Used train-validation-test split of 0.6-0.2-0.2
* Applied data augmentation to test set images: random flipping, affine transformations, rotation, colorjitter and grayscaling
* Calculated pos_weights (to be used with BCEWithLogitsLoss) for all labels using function calculate_label_statistics

### Training
* BCEWithLogitsLoss as the loss function. Used the pos_weight parameter to control the label imbalance in the training data.
* Stochastic gradient descent as the training algorithm
* Batch size x
* Learning rate x
* ?Dropout

### Evaluation
* We used **f1 scores** to evaluate models
  * with **'micro' averaging**, where each sample-class pair is given an equal contribution to the overall metric
* Thresholds

We searched for the optimal threshold by scanning the from 0.05 to 1.0 with 0.05 steps and chose the threshold that led to maximum validation f1 score.

For example, here we would choose 0.75 as the threshold.

![threshold_search.png](images/threshold_search.png)

### Functions and parts of final code
* Data loading
* Visualizing predictions
* Training
* Evaluation of model
* Visualizing confusion matrices for each class


## Final F1-score of the group’s model: xxx

## Other approaches and parameters the group tried
*Let’s describe along with the approach or set of parameters how well it worked.*

### Loading images
Tried ImageFolder for loading images. Noticed that it does not really work for multilabel classification (would work for multiclass single-label classification). After that used adopted the vector of labels per image approach.


### Network type, structure and parameters

#### Simple feedforward network

#### Self-trained convolutional neural networks

#### Transfer learning models
##### VGG16 with pytorch 1.4.0

* replaced output layer with two linear layers 8192->1024 ja 1024->14 and trained it with x epochs
* with threshold=0.75: validation f1 score of 0.73

##### VGG16 with pytorch 1.5.0
* replaced output layer with our own and trained it with x epochs
* validation f1 score of 0.67

##### VGG16 with batch normalization

##### VGG16 without batch normalization

##### ResNet-101

##### ResNet-152


## Other notes
* Labels in the training set are not independent. E.g male and female photos are always also people photos. It's important to train the model with all labels for a certain image so that it can learn from these dependencies.

## Error analysis
Representative examples, with possible explanations about what does not work and how it could be made to work better.

- Difference between male and female labels is in some cases hard to tell (for the network, but also for human observers)
