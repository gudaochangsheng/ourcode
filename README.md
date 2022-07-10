# Description
Our code is built on an open source repository. 
To prevent violating the rules of double-blind review, we have not given a link to the open source repository. 
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




