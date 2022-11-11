## Code for cloing the repository and loding relevant packages in colab.
## Kept here just for reference.
### Important: Run this first!!!! ###
private_key = '''-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW
QyNTUxOQAAACCBQKtLPn9P9fip4iwQ2OO5wa5dPd996k3NIsT10pVvzAAAAKDXmIOS15iD
kgAAAAtzc2gtZWQyNTUxOQAAACCBQKtLPn9P9fip4iwQ2OO5wa5dPd996k3NIsT10pVvzA
AAAEAz7CJey2d2sRRSXEzDTD8RXqq2x2/SNJkEgQB1+loUFIFAq0s+f0/1+KniLBDY47nB
rl09333qTc0ixPXSlW/MAAAAF2Fuc2h1bC5zYXdhbnRAZ21haWwuY29tAQIDBAUG
-----END OPENSSH PRIVATE KEY-----
'''
!mkdir -p /root/.ssh
with open("/root/.ssh/id_ed25519", "w+") as f:
  f.write(private_key)

!ssh-keyscan -t ed25519 github.com > ~/.ssh/known_hosts
# Restrict the key permissions, or else SSH will complain.
!chmod go-rwx /root/.ssh/id_ed25519
!rm -rf /content/Void-Segmentation/
!git clone git@github.com:anshulsawant/Void-Segmentation.git
import tensorflow as tf

from google.colab import drive

import cv2

from glob import glob
import numpy as np
import os

from tensorflow import keras
from tensorflow.keras import layers
from IPython.display import clear_output
import matplotlib.pyplot as plt

from sklearn import metrics as skm
import random

from tensorflow.python.ops import math_ops

SIZE = 1024

import sys
if not '/content/Void-Segmentation/src' in sys.path:
  sys.path.append('/content/Void-Segmentation/src')
import importlib
importlib.invalidate_caches()
import images
import datasets
import utils
importlib.reload(images)
importlib.reload(datasets)
importlib.reload(utils)
