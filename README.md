# Geo-CNN-Pytorch-PYG
A Pytorch Implementation of “Modeling Local Geometric Structure of 3D Point Clouds using Geo-CNN”

- This repository is a reproduction of the GeoCNN, which can support multiple GPUs.
- My environment:
  - Python 3.7
  - Pytorch 1.5.0
  - PYG 1.5.0
  - Cuda 10.2
  - Cudnn 7.6.5

## Accuracy on ModelNet40
|this implementation|original paper|
|---|---|
|93.?|93.4|

## How to Use This Code
- Prepare Data
  - Download [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) data set
  - Move `modelnet40_normal_resampled.zip` into `data/ModelNet40_10000`
  - Unzip `modelnet40_normal_resampled.zip`
  - Rename `modelnet40_normal_resampled` to `raw`
- Train
  - We can change args in the `Configuration` part of the code if you want
  - Then let’s start training: `python geocnn.py`
- Test
  - Uncomment these two lines of code and replace the weight path
  - Comment out this line `# train(epoch)`
  - Then let’s start testing: `python geocnn.py`
