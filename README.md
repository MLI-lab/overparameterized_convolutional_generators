# Denoising and Regularization via Exploiting the Structural Bias of Convolutional Generators

This repository provides code for reproducing the figures in the  paper:

**``Denoising and Regularization via Exploiting the Structural Bias of Convolutional Generators''**, by Reinhard Heckel and Mahdi Soltanolkotabi. Contact: [reinhard.heckel@gmail.com](reinhard.heckel@gmail.com)

The paper is available online [here](http://www.reinhardheckel.com/papers/overparameterized_convolutional_generators.pdf).

## Organization

- Figure 1: denoising_MSE_curves.ipynb
- Figure 4,8: noise_vs_img_fitting_different_architectures.ipynb
- Figure 5: linear_least_squares_selective_fitting_warmup.ipynb
- Figure 6: kernels_and_associated_dual_kernels.ipynb
- Figure 7: Jacobian_multi_layer_deep_decoder.ipynb
- Figure 10: image_fitted_faster_than_noise_on_imgnet.ipynb
- Figure 12: Jacobian_inner_product_noisevsimg.ipynb


## Installation

The code is written in python and relies on pytorch. The following libraries are required: 
- python 3
- pytorch
- numpy
- skimage
- matplotlib
- scikit-image
- jupyter

The libraries can be installed via:
```
conda install jupyter
```

A small part of the code compares performance to the deep image prior. This part requires downloading the models folder from [https://github.com/DmitryUlyanov/deep-image-prior](https://github.com/DmitryUlyanov/deep-image-prior).


## Citation
```
@article{heckel_denoising_2019,
    author    = {Reinhard Heckel and Mahdi Soltanolkotabi},
    title     = {Denoising and Regularization via Exploiting the Structural Bias of Convolutional Generators},
    journal   = {arXiv:?},
    year      = {2019}
}
```

## Licence

All files are provided under the terms of the Apache License, Version 2.0.
