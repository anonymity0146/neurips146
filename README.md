# neurips146

![cf](/Figs/demo.png)


## Requirements

1. The project was implemented and tested in Python 3.5 and Pytorch 1.0. Other versions should work after minor modification.
2. Other common modules like numpy, pandas and cv2 for visualization.
3. NVIDIA GPU and cuDNN are required to have fast speeds. For now, CUDA 8.0 with cuDNN 6.0.20 has been tested. The other versions should be working.

## Implementation details

### data preparation

To reproduce the results on [2] and [3], coco and wikiart datasets should be downloaded to the root path. Details can be accessed in [project page[2]](https://github.com/abhiskk/fast-neural-style) and [project page[3]](https://github.com/sunshineatnoon/LinearStyleTransfer).

### stylization generation

1. To reproduce our basic results and ablation studys,
```
NST_vgg_variant.py
NST_resnet_variant.py
NST_inception.py
NST_wrn.py
```
2. To reproduce comparison results across different architectures based on [2],
```
fast_neural_style_resnet.py
fast_neural_style_resnet_softmax.py
```
3. To reproduce comparison results across different architectures based on [3],
```
Linear_Train_resnet.py
Linear_Train_resnet_softmax.py
Linear_Train_resnet_random.py
Linear_Train_resnet_random_softmax.py
Linear_TestArtistic_resnet.py
```

## References

[1] Gatys, L.A., Ecker, A.S. and Bethge, M., Image style transfer using convolutional neural networks, CVPR, 2016.

[2] Johnson, J., Alahi, A. and Fei-Fei, L., Perceptual losses for real-time style transfer and super-resolution, ECCV, 2016.

[3] Li, X., Liu, S., Kautz, J. and Yang, M.H., Learning linear transformations for fast image and video style transfer, CVPR, 2019.
