import pickle as pkl
import os
import gc
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import scipy as sc
import pandas as pd
import pyEELSMODEL.api as em
import pyEELSMODEL
from pyEELSMODEL.components.linear_background import LinearBG
from pyEELSMODEL.components.CLedge.zezhong_coreloss_edgecombined import ZezhongCoreLossEdgeCombined
from pyEELSMODEL.components.CLedge.kohl_coreloss_edgecombined import KohlLossEdgeCombined
from pyEELSMODEL.components.MScatter.mscatterfft import MscatterFFT
from pyEELSMODEL.components.MScatter.mscatter import Mscatter
from pyEELSMODEL.components.gdoslin import GDOSLin
from pyEELSMODEL.fitters.linear_fitter import LinearFitter
from sklearn.decomposition._nmf import _initialize_nmf as initialize_nmf
from tqdm import tqdm
import hyperspy.api as hs
import colorcet as cc
import shutil

try:
	import cupy as cp
	CUPY_AVAILABLE = True
except:
	CUPY_AVAILABLE = False
	cp = np
	print("cupy not available")



