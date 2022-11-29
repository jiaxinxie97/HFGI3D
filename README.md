# High-fidelity 3D GAN Inversion by Pseudo-multi-view Optimization





[paper](https://arxiv.org/submit/4622938) | [project website](https://ken-ouyang.github.io/HFGI3D/index.html)
  
<img src="pics/teaser.png" width="800px"/> 

## Introduction
We present a high-fidelity 3D generative adversarial net-
work (GAN) inversion framework that can synthesize photo-
realistic novel views while preserving specific details of the
input image.

<img src="pics/method.png" width="800px"/>  


## Set up
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

### More Results
Video results are shown on our [project website](https://ken-ouyang.github.io/HFGI3D/index.html).

### Acknowlegement   
We thank the authors of [EG3D](https://github.com/NVlabs/eg3d) and [PTI](https://github.com/danielroich/PTI) for sharing their code.

### Citation
If you find this work useful for your research, please cite:



