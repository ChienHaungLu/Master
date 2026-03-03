import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84))
    frame = frame.astype(np.float32) / 255.0
    return frame