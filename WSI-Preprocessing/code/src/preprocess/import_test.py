# test_import.py
import time

start = time.time()

import numpy
print("numpy:", time.time()-start)

start = time.time()
import cv2
print("cv2:", time.time()-start)

start = time.time()
from openslide import OpenSlide
print("openslide:", time.time()-start)

start=time.time()
import matplotlib.pyplot as plt
print("matplotlib", time.time()-start)

start=time.time()
from sklearn.cluster import KMeans
print("sklearn", time.time()-start)

start=time.time()
from skimage import morphology
print("skimage", time.time()-start)

start=time.time()
from scipy.ndimage import binary_fill_holes
print("scipy", time.time()-start)

start=time.time()
from tiatoolbox.tools.tissuemask import OtsuTissueMasker
print("tiatoolbox", time.time()-start)

start=time.time()
import pandas as pd
print("pandas", time.time()-start)

start=time.time()
import seaborn as sns
print("seaborn", time.time()-start)

start = time.time()
from src.preprocess.masking.week2 import double_pass_and_p1wr
print("week2:", time.time()-start)

start = time.time()
from src.preprocess.tiling.optimized_tiler_vips import MemorySafeTiler
print("tiler:", time.time()-start)

start = time.time()
import config
print("config:", time.time()-start)

start = time.time()
import matplotlib.pyplot as plt
print("matplotlib:", time.time()-start)

