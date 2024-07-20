import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import center_of_mass
from math import *
from PIL import Image
import os
from torchvision import transforms 
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
