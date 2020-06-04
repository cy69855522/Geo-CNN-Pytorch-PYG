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
  - Rename `modelnet40_normal_resampled.zip` to `raw`
  - Remove `modelnet40_normal_resampled.zip`
- Train
  - We can change args in the `Configuration` part of the code
  - Then Let’s start training: `python geocnn.py`
- Test
  - Uncomment the following two lines of code and replace the weight path
  
    '''python
    #model.load_state_dict(torch.load('geocnn_epoch_235_0.9323338735818476', map_location=f'cuda:{device_list[0]}'), strict=True)
    #optimizer.load_state_dict(torch.load('geocnn_optimizer.pt', map_location=f'cuda:{device_list[0]}').state_dict())
    '''
  - Comment out this line `#train(epoch)`
  - Then Let’s start testing: `python geocnn.py`
