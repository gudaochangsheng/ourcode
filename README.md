# Description
Our code is built on an open source repository[OpenGait](https://github.com/ShiqiYu/OpenGait). This is the code for the paper [GaitMM: Multi-Granularity Motion Sequence Learning for Gait Recognition](https://arxiv.org/abs/2209.08470).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gaitfm-fine-grained-motion-representation-for/gait-recognition-on-oumvlp)](https://paperswithcode.com/sota/gait-recognition-on-oumvlp?p=gaitfm-fine-grained-motion-representation-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gaitfm-fine-grained-motion-representation-for/multiview-gait-recognition-on-casia-b)](https://paperswithcode.com/sota/multiview-gait-recognition-on-casia-b?p=gaitfm-fine-grained-motion-representation-for)

# Operating Environments
## Hardware Environment
Our code is running on a server with 2 GeForce RTX 3090 GPUs 
and a CPU model Intel(R) Core(TM) i7-9800X @ 3.80GHz.
## Software Environment
- pytorch = 1.10
- torchvision
- pyyaml
- tensorboard
- opencv-python
- tqdm

# Code with Comments
To make the code easier to read, we have collated the model code and added comments.
The model code is [lib/modeling/models/GaitFM.py](lib/modeling/models/GaitFM.py).
We also give the configuration file code for the training model.
## For the CASIA-B dataset
The configuration file is [config/gaitfm.yaml](config/gaitfm.yaml).
## For the OUMVLP dataset
The configuration file is [config/gaitfm_OUMVLP.yaml](config/gaitfm_OUMVLP.yaml).

# Citation
```
@inproceedings{wang2023gaitmm,
  title={GAITMM: Multi-Granularity Motion Sequence Learning for Gait Recognition},
  author={Wang, Lei and Liu, Bo and Wang, Bincheng and Yu, Fuqiang},
  booktitle={2023 IEEE International Conference on Image Processing (ICIP)},
  pages={845--849},
  year={2023},
  organization={IEEE}
}
```




