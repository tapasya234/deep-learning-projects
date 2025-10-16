import numpy as np
import requests
import torchvision
import torch
import zipfile
import matplotlib.pyplot as plt
import glob as glob
import os
import ast
from PIL import Image
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from data_path import DATA_PATH
