from SUAVE.Core import Units, Data
import numpy as np

#import Plot_Mission
import SUAVE.Optimization.Package_Setups.scipy_setup as scipy_setup
import SUAVE.Optimization.Package_Setups.pyopt_setup as pyopt_setup

from pyKriging import saveModel, loadModel
from pyKriging.krige import kriging  

#from SUAVE.Attributes.Solids import Aluminium, Bidirectional_Carbon_Fiber, Unidirectional_Carbon_Fiber
import csv, datetime, os

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from matplotlib.ticker import LinearLocator, FormatStrFormatter



[cand10,gen10] = loadModel('k0-30km2-lhc10-spantpsw-_optcands_generations.pkl')
[cand20,gen20] = loadModel('k0-30km2-spantpsw--lhc20_optcands_generations.pkl')
[cand30,gen30] = loadModel('k0-30km2-spantpsw--lhc30_optcands_generations.pkl')
[cand40,gen40] = loadModel('k0-30km2-spantpsw--lhc40_optcands_generations.pkl')
[cand50,gen50] = loadModel('k0-30km2-spantpsw--lhc50_optcands_generations.pkl')

dat10 = np.array(gen10[-1])
dat20 = np.array(gen20[-1])
dat30 = np.array(gen30[-1])
dat40 = np.array(gen40[-1])
dat50 = np.array(gen50[-1])

dat = [dat10,dat20,dat30,dat40,dat50]
num = ['10','20','30','40','50']


fig,ax = plt.subplots()
ax.grid()
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.xaxis.grid(color='gray', linestyle='dashed')
cols = cm.brg(np.linspace(0,1,5))

for i in range(0,len(dat)):
    print num[i]
    aset= dat[i]
    xi  = -aset[:,0]
    yi  = aset[:,1]
    sc=ax.scatter(xi,yi,marker='o',c=cols[i],s=12,alpha=1.)

cax, _ = matplotlib.colorbar.make_axes(ax)
cmap = matplotlib.cm.get_cmap('brg')
normalize = matplotlib.colors.Normalize(vmin=10,vmax=50)
cbar = matplotlib.colorbar.ColorbarBase(cax,cmap=cmap,norm=normalize)
cbar.set_label('LHC size',rotation=270)
ax.set_title('Pareto convergence - Correlation')
ax.set_xlabel('L/D')
ax.set_ylabel('Mass (kg)')
plt.show()





