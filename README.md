# BRIDE-Fine tuning-HTC-2022

***Boundary Refinement of Intrinsic Deblurring Estimation*** for Helsinki Tomography Challenge.

<font size=3> Wang Jianyu<sup>1</sup>, Wang Rongqian<sup>1</sup>, Liu Xintong<sup>1</sup>, Lin Guochang<sup>1</sup>, Chen Fukai<sup>1</sup>, Cai Lide<sup>2</sup> </font>

<font size=2><sup>1</sup> Yau Mathematical Sciences Center, Tsinghua University, Beijing, China</font>

<font size=2> <sup>2</sup> Department of Mathematical Science, Tsinghua University, Beijing, China </font>

## 1. The limited-angle CT challenge

The [Helsinki Tomography Challenge 2022(HTC 2022)](https://www.fips.fi/HTC2022.php) is about limited-angle computational tomography. The main challenge of this image reconstruction problem are as follows

- Due to the extremely limited probing degree, the obtained real-life data inevitably miss the information of the wavefront set of the singular support of the obstacle. This part of theory is completed by Todd Quinto using microlocal analysis since the late 1980s. It shows the incapability of the back-projection method, even if the measurement is dealt with carefully.
- Due to the ill-posedness induced by the loss of information, measurement error is amplified tremendously in the inversion process, thus suitable regularization technique should be taken. 

## 2. A brief introduction of the BRIDE algorithm

Concerning the above challenges, we propose an algorithm which combines the back-projection(BP) method, image deblurring network, image super resolution network and the Bayesian inference method. A brief introduction is provided below and we will discuss the method in detail in the future.

- **Step 1: Back projection** 

  > The measured limited-angle sinogram is first processed by the BP method. In this part, the implement of the BP method is based on the `astra` package.

- **Step 2: Deblurring**

  > The reconstruction of the BP method is then passed to a deblurring network to recover the details of the phantom. The code of this network is based on *[Deep Residual Fourier Transformation for Single Image Deblurring](https://github.com/INVOKERer/DeepRFT)*. We modify and train the network to improve the quality of the reconstruction. The reconstruction process can be viewed as the signal processed with the adjoint operator followed by an inverse Gram operator.

+ **Step 3: Super resolution**

  > The challenge requires that the output figures be of size 512 $\times$ 512, which poses a challenge on the training of the neural network. To tackle this problem, we train the network using images with 128 $\times$ 128 pixels and apply a super resolution network to obtain figures with 512 $\times$ 512 pixels. This super resolution network is based on *[`Residual Dense Network for Image Super-Resolution`](https://github.com/yulunzhang/RDN)*.

- **Step 4: Boundary Refinement** 

  > We term the output of step 3 the `Intrinsic Deblurring Estimation`. This estimation helps to construct the prior of the final reconstruction. In this last step, the boundary of the target is refined by minimizing a combination of the data fidelity term and the regularization term.

- **Step 5: Fine-tuning**: The reconstuction is fine tuned by a generative model based on *[Denoising Diffusion Probabilistic Models](https://github.com/INVOKERer/DeepRFT)*. 

- **Step 6: Thresholding**: The result is obtained by applying a threshold to the high-resolution image. 

## 3. Installation instructions

### 3.1 Python packages

The code uses the following python packages.

```
argparse
astra
math
numpy
os
pylab
scipy
skimage
torch
matlab
```

To install these python packages(except matlab), run:

```
cd ./installation
conda env create -f Torch.yaml
```

To call matlab functions in python, we refer to https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html?requestedDomain=us.

### 3.2 Matlab toolboxes

The code uses the following matlab toolboxes.

```
ASTRA Tomography Toolbox
HelTomo Toolbox
Spot Linear-Operator Toolbox
```

To install these toolboxes for Windows, download the `toolbox.rar` from https://cloud.tsinghua.edu.cn/d/5bd4c08fd94641ce981a/ and unzip it in the main folder.

## 4. Usage instructions

To test the algorithm, just run the following code

```python
python main.py --data_dir './data/' --out_dir './output/' --group_number 7
```

`--data_dir` : The folder in which the input files are located.

`--out_dir` : The folder in which the output files are stored.

`--group_number` : Group category number, from 1 to 7.

## 5. An example

Now run the following code

```
python main.py --group_number 7
```

and check the `output` folder. The reconstruction is `test_data.png`

**Ground truth**:

<img src="README.assets/htc2022_ta_full_recon_fbp_seg.png" alt="gt.png" style="zoom:33%;" />

**Results**:

| Opening angle |                              BP                              |                             Ours                             |
| :-----------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|      30°      | <img src="README.assets/image-20221026193249277.png" alt="BP.png" style="zoom:33%;" /> | <img src="README.assets/test_data.png" alt="ours.png" style="zoom:33%;" /> |

## tl;dr

Install the python packages.

```
cd ./installation
conda env create -f Torch.yaml
```

Download and unzip `toolbox.rar` from https://cloud.tsinghua.edu.cn/d/5bd4c08fd94641ce981a/.

Run the python code.

```
python main.py --data_dir './data/' --out_dir './output/' --group_number 7
```


