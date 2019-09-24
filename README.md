# Auto-colorization-for-greyscale-images
Neural networks for colorizing black &amp; white photos automatically
### Brief Description

The code in this program is a reproduction of the following article:

<https://blog.floydhub.com/colorizing-b-w-photos-with-neural-networks/>

The goal is to colorize black & white photos automatically.

First, I use a specific variety of cat -- Siamese kitten, to train the beta-version neural network proposed in the above article. I use 200 images to train this 16-layers neural network. The results seems acceptable and reasonable.

Second, I use different kinds of cat & dog images to train the full-version neural network proposed in the above article, which fuses the classification networks output with the encoder results then input decoder together. The classifier is "the inception resnet V2". This enables the network to match an object representation with a colouring scheme, so image features can be better understood. So I can input training images containing different varieties of cats, dogs and other objectives, let the network classifies, and I can still get a acceptable results. And this time, considering my computer's limited memory space and GPU performance, I only choose 60 images for training. 



### Catalogue

##### code:

colornet_beta.py: train and test for model without classification network.

colornet_final.py: train and test for model with classification network.

preprocess.py: input images pre-procession.

##### folder:

weights: inception resnet V2 weights

output: output images (not shown) & .h5 model

pics: training data (not shown)



### Structure
waiting for...


### Data

##### Database

- Kaggle Dog vs. Cat database（select 60 images for train，100 images for test）
- google: Siamese kitten, bulk download by Chrome extensions: Fatkun (200 images for train, 30 images for test)

##### Pre-process

Zoom the training images to 256*256 pixel size, using black pixels as padding.
