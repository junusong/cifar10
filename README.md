# Readme 

The purpose of this project was to classify images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).  
To achieve this, I built a 2-layer Convolutional Neural Network (CNN) in PyTorch.
Although PyTorch has the CIFAR-10 dataset already loaded and preprocessed for use under `torch.torchvision`, I was specifically tasked to immplement a preprocessing pipeline separately.

### Citations 
Because I have had no prior experience with PyTorch or CNNs, many online resources were used to help me through this project. 

#### Youtube videos for theory 
- [Neural Networks video series by StatQuest with Josh Starmer](https://www.youtube.com/watch?v=CqOfi41LfDw&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1)
- [Recorded MIT lecture on CNNs by Alexander Amini](https://www.youtube.com/watch?v=iaSUYvmCekI)
- [Short video on CNNs by deeplizard](https://www.youtube.com/watch?v=YRhxdVk_sIs)

#### PyTorch official website 
- [Training a Classifier CIFAR-10 official tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [PyTorch Dataset tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [Source code for torchvision.datasets.cifar](https://pytorch.org/vision/0.12/_modules/torchvision/datasets/cifar.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

#### Other sites for various other help 
- [Medium publication on how to preprocess CIFAR-10 data](https://medium.com/@rhythm10/image-preprocessing-for-cifar-10-dataset-f2b5cdb221bb)
- [Custom CIFAR-10 implemenation with PyTorch](https://www.kaggle.com/code/uvxy1234/cifar-10-implementation-with-pytorch/notebook) - referenced for handling the preprocessing 
- [Matplotlib introduction on how to plot graphs](https://matplotlib.org/stable/tutorials/introductory/pyplot.html)
- ["Accuracy, Precision, Recall or F1?" by Koo Ping Shung](https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9)
- ["A Gentle Introduction to Cross-Entropy for Machine Learning" by Jason Brownlee](https://machinelearningmastery.com/cross-entropy-for-machine-learning/) (however, I can't say I completely understand cross-entropy)

Stackoverflow links for specific issues can be found in the inline commens of the source code. 
