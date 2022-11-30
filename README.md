# High-fidelity 3D GAN Inversion by Pseudo-multi-view Optimization





[paper](https://arxiv.org/abs/2211.15662) | [project website](https://ken-ouyang.github.io/HFGI3D/index.html)
 
<img src="pics/input01.png" width="230px"/>        <img src="pics/input_video01.gif" width="230"/>  <img src="pics/input02.png" width="230px"/>        <img src="pics/input_video02.gif" width="230"/> 

<img src="pics/input03.png" width="230px"/>        <img src="pics/input_video03.gif" width="230"/>  <img src="pics/input04.png" width="230px"/>        <img src="pics/input_video04.gif" width="230"/> 

<img src="pics/input07.png" width="230px"/>        <img src="pics/input_video07.gif" width="230"/>  <img src="pics/input09.png" width="230px"/>        <img src="pics/input_video09.gif" width="230"/> 
 

## Introduction
We present a high-fidelity 3D generative adversarial net-
work (GAN) inversion framework that can synthesize photo-
realistic novel views while preserving specific details of the
input image.

<img src="pics/method.png" width="800px"/>


## Setup
### Installation
```
git clone https://github.com/jiaxinxie97/HFGI3D.git
cd HFGI3D
```

### Environment

```
conda env create -f environment.yml
conda activate HFGI3D
```

## To Do
- [x] Release the pose estimation code for customized images
- [x] Release the editing code

## Quick Start

### Prepare Images
We put some examples images and their corresponding pose in `./test_imgs`, also we put the configs files of examples in `./example_configs/`, and you can quickly try them.   
For customized images, it is encouraged to first pre-process (align & crop) and extract pose for them, and then inverse and edit with our model. Code for this part will be released soon.

### Pretraind model
Download the pretrained generator on FFHQ from [EG3D](https://github.com/NVlabs/eg3d). For convenience, we upload it in [Google drive](https://drive.google.com/file/d/1rsF-IHBLW7WvDckdbNK9Qm9SwHK02E5l/view?usp=sharing). Download  `ffhq512-128.pkl` and put it in `./inversion/`.

### Optimization
  
```
cd inversion/scripts
python run_pti.py ../../example_configs/config_00001.py
```

## More Results
Video results are shown on our [project website](https://ken-ouyang.github.io/HFGI3D/index.html).

## Acknowlegement   
We thank the authors of [EG3D](https://github.com/NVlabs/eg3d) and [PTI](https://github.com/danielroich/PTI) for sharing their code.





