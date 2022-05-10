import sys
import os
import time
import math
import numpy as np

particle_radius = 0.5
strObj = "T"
strGen = "python GenVoxel.py  -i " + strObj + " -s " + str(particle_radius)
#strDis = "python ShowVoxel.py -i " + strObj + " -r " + str(particle_radius)

### Generate Voxel
os.system(strGen)
#os.system(strDis)



