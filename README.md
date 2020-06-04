# Geo-CNN-Pytorch-PYG
A Pytorch Implementation of “Modeling Local Geometric Structure of 3D Point Clouds using Geo-CNN”

- This repository is a reproduction of the GeoCNN, which can support multiple GPUs.
- My enviroment:
  - Ubuntu 18.04
  - Python 3.7
  - Pytorch 1.5.0
  - PYG 1.5.0
  - Cuda 10.2
  - Cudnn 7.6.5
  - GPU Memory >= 8G

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
  - We can change args in the [Configuration part](https://github.com/cy69855522/Geo-CNN-Pytorch-PYG/blob/master/geocnn.py#L25) of the code if you want
  - Then let’s start training: `python geocnn.py`
- Test
  - Uncomment [this line](https://github.com/cy69855522/Geo-CNN-Pytorch-PYG/blob/master/geocnn.py#L248) and replace the weight path
  - Set [only_test](https://github.com/cy69855522/Geo-CNN-Pytorch-PYG/blob/master/geocnn.py#L40) as `True`
  - Then let’s start testing: `python geocnn.py`
