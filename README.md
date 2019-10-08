# ACMMM2018-HADDA

Pytorch code for Feature Constrained by Pixel: Hierarchical Adversarial Deep Domain Adaptation</a> in ACM MM 2018

The framework of the proposed method:

![image](https://github.com/jimmykobe/ACMMM2018-HADDA/blob/master/models/acmmm2018.png "image")

# Setup

* Prerequisites: Python3.6, pytorch=0.3.0, Numpy, Pillow, SciPy

* The source code files:

  1. "models": Contains the network architectures and the definitions of the loss functions.
  2. "core": Contains the pratraining, training files.
  3. "datasets": Contains datasets loading
  4. "misc": Contains initialization and some preprocessing functions
  

# Training

To run the main file with the file config_MU or config_PIE2705 in misc folder

# Acknowledge
Please cite the paper:

Please kindly cite this paper in your publications if it helps your research:
```
@inproceedings{mm/ShaoLY18,
  author    = {Rui Shao and
               Xiangyuan Lan and
               Pong C. Yuen},
  title     = {Feature Constrained by Pixel: Hierarchical Adversarial Deep Domain
               Adaptation},
  booktitle = {2018 {ACM} Multimedia Conference on Multimedia Conference, {MM} 2018,
               Seoul, Republic of Korea, October 22-26, 2018},
  pages     = {220--228},
  year      = {2018},
```

Contact: ruishao@comp.hkbu.edu.hk
