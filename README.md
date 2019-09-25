# Auto-colorization-for-greyscale-images
Neural networks for colorizing black &amp; white photos automatically


## Brief Description

The code in this program is a reproduction of the following article:

<https://blog.floydhub.com/colorizing-b-w-photos-with-neural-networks/>

The goal is to colorize black & white photos automatically.

First, I use a specific variety of cat -- Siamese kitten, to train the beta-version neural network proposed in the above article. I use 200 images to train this 16-layers neural network. The results seems acceptable and reasonable.

Second, I use different kinds of cat & dog images to train the full-version neural network proposed in the above article, which fuses the classification networks output with the encoder results then input decoder together. The classifier is "the inception resnet V2". This enables the network to match an object representation with a colouring scheme, so image features can be better understood. So I can input training images containing different varieties of cats, dogs and other objectives, let the network classifies, and I can still get a acceptable results. And this time, considering my computer's limited memory space and GPU performance, I only choose 60 images for training. 



## Catalogue

#### code:

colornet_beta.py: train and test for model without classification network.

colornet_final.py: train and test for model with classification network.

preprocess.py: input images pre-procession.

#### folder:

final: part of output pictures

structure: the structure introduction of two version networks


## Structure
#### Beta Version

![image](https://github.com/JJHAirylin/Auto-colorization-for-greyscale-images/blob/master/structure/beta_version.png)

*Highlight*: 

- using 2 strides conv filter to replace max-pooling, in case of distoration
- activation function: relu, except for output layer: tanh
- In the `ImageDataGenerator`, we adjust the setting for our image generator. This way, one input image will never be the same.
- colorspace: Lab 
- Loss: mean square error (1000 epoch final average loss: 8e-4)
- model size: 29.7M


#### Final Version

![image](https://github.com/JJHAirylin/Auto-colorization-for-greyscale-images/blob/master/structure/final_version.png)

*Highlight:*

- split whole network into four parts: encoder, classifier, fusion layer and decoder
- add classification network output (1\*1\*1000) to encoder output (32\*32\*256), and together as the input of decoder
- loss: mean square error (1000 epoch final average loss: about 0.0030)
- model size: 74.9M


## Results
#### Beta Version -- Siamese kitten colorization 
At first, I train this network with 500 images of different variety of cats&dogs for 100 epoch. 

The test outputs truned out bad, the whole picture is in brown.

Input -> Output -> Ground Truth:

![image](https://github.com/JJHAirylin/Auto-colorization-for-greyscale-images/blob/master/final/normal.png)


When this network just learn the pictures of siamese kitten, test results become better:

![image](https://github.com/JJHAirylin/Auto-colorization-for-greyscale-images/blob/master/final/xianluo_1.png)
![image](https://github.com/JJHAirylin/Auto-colorization-for-greyscale-images/blob/master/final/xianluo_2.png)
![image](https://github.com/JJHAirylin/Auto-colorization-for-greyscale-images/blob/master/final/xianluo_3.png)
![image](https://github.com/JJHAirylin/Auto-colorization-for-greyscale-images/blob/master/final/xianluo_4.png)
![image](https://github.com/JJHAirylin/Auto-colorization-for-greyscale-images/blob/master/final/xianluo_5.png)
![image](https://github.com/JJHAirylin/Auto-colorization-for-greyscale-images/blob/master/final/xianluo_6.png)


It shows particular features like blue eyes, brown neck, green or blue background.

But if this network is used on generalized cats and dogs colorization, the results are poor even sometimes ridiculous:

![image](https://github.com/JJHAirylin/Auto-colorization-for-greyscale-images/blob/master/final/xianluo_7.png)
![image](https://github.com/JJHAirylin/Auto-colorization-for-greyscale-images/blob/master/final/xianluo_8.png)


#### Final Version -- Extra Classifier 
After adding extra classifier to beta version network, this mixed-network should enable to identify more objects hence becoming more generic.

I train this network with only 60 images of cats&dogs for 1000 epoch. 

The test set is comprised of 100 images, most of them perform badly (just brown covers a large area on animals), but I still can choose some good results from them.

![image](https://github.com/JJHAirylin/Auto-colorization-for-greyscale-images/blob/master/final/final_1.png)
![image](https://github.com/JJHAirylin/Auto-colorization-for-greyscale-images/blob/master/final/final_2.png)
![image](https://github.com/JJHAirylin/Auto-colorization-for-greyscale-images/blob/master/final/final_3.png)
![image](https://github.com/JJHAirylin/Auto-colorization-for-greyscale-images/blob/master/final/final_4.png)
![image](https://github.com/JJHAirylin/Auto-colorization-for-greyscale-images/blob/master/final/final_5.png)
![image](https://github.com/JJHAirylin/Auto-colorization-for-greyscale-images/blob/master/final/final_8.png)
![image](https://github.com/JJHAirylin/Auto-colorization-for-greyscale-images/blob/master/final/final_9.png)


It seems like it is goot at colorizing lawn and grass. But unfortunately it sometimes mistakes animal fur for grass:

![image](https://github.com/JJHAirylin/Auto-colorization-for-greyscale-images/blob/master/final/final_6.png)
![image](https://github.com/JJHAirylin/Auto-colorization-for-greyscale-images/blob/master/final/final_7.png)


## Data

#### Database

- Kaggle Dog vs. Cat database（select 60 images for train，100 images for test）
- google: Siamese kitten, bulk download by Chrome extensions: Fatkun (200 images for train, 30 images for test)


#### Pre-process

Zoom the training images to 256*256 pixel size, using black pixels as padding.
