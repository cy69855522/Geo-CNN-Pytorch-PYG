# Geo-CNN-Pytorch-PYG
A Pytorch Implementation of “Modeling Local Geometric Structure of 3D Point Clouds using Geo-CNN”

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
  - We can change args in the `Configuration` part of the code
  - Then Let’s start training: `python geocnn.py`
- Test
  - Uncomment these two lines of code and replace the weight path
  - Comment out this line `# train(epoch)`
  - Then Let’s start testing: `python geocnn.py`
