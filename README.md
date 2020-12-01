# DoGAN

An experiment to generate dog images based on a dataset on Kaggle. Planning to do this with an architecture called Generative Adversarial Networks or GAN. The library used will be PyTorch, a very effective and easy to use deep learning library.

Look at how cute she is!
<img src='https://upload.wikimedia.org/wikipedia/commons/5/51/Lucy_the_Dog_at_The_Green%2C_Town_Square_Las_Vegas.jpg'>

## The Data

The data will be simple 100x100 images of dogs found off Kaggle. These images were originally different sizes, but a simple script was used to resize and save all of the images. There are 1000 images of dogs in this dataset.

## The GAN

The GAN will be a WGAN-GP, with the critic outputting a score rather than a binary output indicating real or fake.
The score outputted by the critic should be higher for real images and lower for fake images (the score includes negative numbers).
The loss function will have a gradient penalty term which was shown to be much better (in terms of stability and trainability) compared to the original WGAN.

## The Critic

The critic will be your typical CNN with a linear activation at the last layer. The linear activation is used so that the critic can output a score with any range.

## The Generator

The generator will have an input vector of size 512 and output a 100x100 image. It uses Convolutional Transpose 2D layers rather than using an upsample + Convolutional 2D layers. The final layer has a tanh activation, as we want images with a range of [-1, 1].

## Current State

So far, I've managed to make the network imitate a red square, which can be found in the example_data folder. Imitating dog images has proven to be much more difficult D':. But, I'll keep trying!