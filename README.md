# ImageColorizationUsingPatchGAN

A PyTorch implementation of PatchGAN for image colorization.

## Related Papers

Kamyar Nazeri, Eric Ng and Mehran Ebrahimi:
Image Colorization with Generative Adversarial Networks.
https://arxiv.org/pdf/1803.05400.pdf

Phillip Isola, Jun-Yan Zhu, Tinghui Zhou and Alexei A. Efros:
Image-to-Image Translation with Conditional Adversarial Networks.
https://arxiv.org/pdf/1611.07004.pdf

## Training

1. Clone this repository and move to the directory.

```shell
% clone https://github.com/mps-research/ImageColorizationUsingPatchGAN.git
% cd ImageColorizationUsingPatchGAN
```

2. Download "places365standard_easyformat.tar" from [Places365 Dataset web page](http://places2.csail.mit.edu/index.html).

3. Put the tar into the data directory and extract files.

```shell
% cd data
% tar xvf places365standard_easyformat.tar
```

4. At the repository root directory, build "pgan" docker image and run the image inside of a container.

```shell
% docker build -t pgan .
% ./train.sh
```

## Checking Training Results

At the repository root directory, execute the following command.

```shell
% ./run_tensorboard.sh
```
