# BRIDE-Fine tuning-HTC-2022

***Boundary Refinement of Intrinsic Deblurring Estimation*** for Helsinki Tomography Challenge.

<font size=3> Wang Jianyu<sup>1</sup>, Wang Rongqian<sup>1</sup>, Liu Xintong<sup>1</sup>, Lin Guochang<sup>1</sup>, Chen Fukai<sup>1</sup>, Cai Lide<sup>2</sup> </font>

<font size=2><sup>1</sup> Yau Mathematical Sciences Center, Tsinghua University, Beijing, China</font>

<font size=2> <sup>2</sup> Department of Mathematical Science, Tsinghua University, Beijing, China </font>

## The limited-angle CT challenge

The [Helsinki Tomography Challenge 2022(HTC 2022)](https://www.fips.fi/HTC2022.php) is about limited-angle computational tomography. The main challenge of this image reconstruction problem are as follows

- Due to the extremely limited probing degree, the obtained real-life data inevitably miss the information of the wavefront set of the singular support of the obstacle. This part of theory is completed by Todd Quinto using microlocal analysis since the late 1980s. It shows the incapability of Back-projection method, even if the measurement is dealt with carefully.
- Due to the ill-posedness induced by the loss of information, measurement error is amplified tremendously in the inversion process, thus suitable regularization technique should be taken. 

## Algorithm Introduction
Concerning the above challenges, we propose an algorithm which is a combination of back-projection(BP) method and image deblurring network. To meet the requirements of the competition, the proposed algorithm consists of the following steps:
- **BP**: The measured limited-angle sinogram is first processed by the BP method. In this part, the implement of the BP method is based on astra package.
- **Deblur**: The reconstruction of the BP method is then passed to a deblurring network to recover the details of the phantom. The code of this network is based on *[Deep Residual Fourier Transformation for Single Image Deblurring](https://github.com/INVOKERer/DeepRFT)*. We modify and train the network to improve the quality of the reconstruction.
- **Super-resolution**: Our networks are trained with the size of 128\*128. We use super-resolution to upsample the results to 512\*512. The network is based on *[Residual Dense Network for Image Super-Resolution](https://github.com/yulunzhang/RDN)*.
- **Boundary Refinement**: The output of the previous steps provides an estimation of the target, and is used as part of the prior term. The boundary of the target can be refined by minimizing the data fidelity term and the joint regularization terms of the signal and the object. This idea comes from the paper *[Non-line-of-sight reconstruction with signal–object collaborative regularization](https://www.nature.com/articles/s41377-021-00633-3)*.
- **Fine-tuning**: The reconstuction is fine tuned by a generative model based on *[Denoising Diffusion Probabilistic Models](https://github.com/INVOKERer/DeepRFT)*. 
- **Thresholding**: The result is obtained by applying a threshold to the high-resolution image. 

## Installation instructions

The code uses the following packages:

```
for python:
argparse
astra
math
matlab
numpy
os
pylab
scipy
skimage
torch


for matlab:
ASTRA Tomography Toolbox
HelTomo Toolbox
Spot Linear-Operator Toolbox
```

To install these packages(except matlab), run:

```
cd ./installation
conda env create -f Torch.yaml
```

To call matlab functions in python, we refer to https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html?requestedDomain=us.

## Usage instructions

The pre-trained weights of the networks and part of the training data of HTC 2022 are provided in this repository. To test the algorithm, just run the following code

```
python main.py --data_dir './data/' --out_dir './output/' --group_number 1
```

`--data_dir` : Folder where the input files are located.

`--out_dir` : Folder where the output files is stored.

`--group_number` : Group category number.

## Examples

We list the results of `htc2022_ta_full` here as an example. The test data whose projection angle is limited to [0,30] is provided in the data folder as an example. The result can be easily obtained by running

```
python main.py --group_number 7
```

**Ground truth**:

<img src="README.assets/gt.png.png" alt="gt.png" style="zoom:33%;" />

**Results**:

| Opening angle |                              BP                              |                             Ours                             |
| :-----------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|      30°      | <img src="README.assets/BP.png.png" alt="BP.png" style="zoom:33%;" /> | <img src="README.assets/deblur_ccsocr_ddpm_ta_30.png" alt="ours.png" style="zoom:33%;" /> |



