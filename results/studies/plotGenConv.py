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
[cand30,gen30] = loadModel('/home/ashaiden/Documents/surrmodthesis/model/rawresults/kriging/k0-spantpsw-(37, 3)_optcands_generations.pkl')
[cand40,gen40] = loadModel('k0-30km2-spantpsw--lhc40_optcands_generations.pkl')
#[cand50,gen50] = loadModel('k0-30km2-spantpsw--lhc50_optcands_generations.pkl')

datck9 = np.array(gen10[-1])
datck11 = np.array(gen20[-1])
dat30 = np.array(gen30[-1])
datck17 = np.array(gen40[-1])
#dat50 = np.array(gen50[-1])

#dat = [dat10,dat20,dat30,dat40,dat50]
#num = ['10','20','30','40','50']


#fig,ax = plt.subplots()
#ax.grid()
#ax.set_axisbelow(True)
#ax.yaxis.grid(color='gray', linestyle='dashed')
#ax.xaxis.grid(color='gray', linestyle='dashed')
#cols = cm.brg(np.linspace(0,1,5))

#for i in range(0,len(dat)):
#    print num[i]
#    aset= dat[i]
#    xi  = -aset[:,0]
#    yi  = aset[:,1]
#    sc=ax.scatter(xi,yi,marker='o',c=cols[i],s=12,alpha=1.)

#cax, _ = matplotlib.colorbar.make_axes(ax)
#cmap = matplotlib.cm.get_cmap('brg')
#normalize = matplotlib.colors.Normalize(vmin=10,vmax=50)
#cbar = matplotlib.colorbar.ColorbarBase(cax,cmap=cmap,norm=normalize)
#cbar.set_label('LHC size',rotation=270)
#ax.set_title('Pareto convergence - Correlation')
#ax.set_xlabel('L/D')
#ax.set_ylabel('Mass (kg)')
#plt.show()

#[cand10,gen10] = loadModel('/home/ashaiden/Documents/surrmodthesis/results/studies/k2-spantpsw--(17, 3)_optcands_generations.pkl')
#[cand20,gen20] = loadModel('/home/ashaiden/Documents/surrmodthesis/results/studies/k2-spantpsw-(22, 3)_optcands_generations.pkl')
#[cand30,gen30] = loadModel('/home/ashaiden/Documents/surrmodthesis/results/studies/k2-spantpsw-(27, 3)_optcands_generations.pkl')
#[cand40,gen40] = loadModel('/home/ashaiden/Documents/surrmodthesis/results/studies/k2-spantpsw-(32, 3)_optcands_generations.pkl')
[cand50,gen50] = loadModel('/home/ashaiden/Documents/surrmodthesis/results/studies/k2-spantpsw-(37, 3)_optcands_generations.pkl')

#dat10 = np.array(gen10[-1])
#dat20 = np.array(gen20[-1])
#dat30 = np.array(gen30[-1])
#dat40 = np.array(gen40[-1])
dat50 = np.array(gen50[-1])

#dat = [dat10,dat20,dat30,dat40,dat50]
#num = ['17','22','27','32','37']


#fig,ax = plt.subplots()
#ax.grid()
#ax.set_axisbelow(True)
#ax.yaxis.grid(color='gray', linestyle='dashed')
#ax.xaxis.grid(color='gray', linestyle='dashed')
#cols = cm.brg(np.linspace(0,1,5))

#for i in range(0,len(dat)):
#    print num[i]
#    aset= dat[i]
#    xi  = -aset[:,0]
#    yi  = aset[:,1]
#    sc=ax.scatter(xi,yi,marker='o',c=cols[i],s=12,alpha=1.)

#cax, _ = matplotlib.colorbar.make_axes(ax)
#cmap = matplotlib.cm.get_cmap('brg')
#normalize = matplotlib.colors.Normalize(vmin=17,vmax=37)
#cbar = matplotlib.colorbar.ColorbarBase(cax,cmap=cmap,norm=normalize)
#cbar.set_label('LHC size',rotation=270)
#ax.set_title('Pareto convergence - Correlation')
#ax.set_xlabel('L/D')
#ax.set_ylabel('Mass (kg)')
#plt.show()

fig,ax = plt.subplots()
ax.grid()
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.xaxis.grid(color='gray', linestyle='dashed')

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

xck9 = -datck9[:,0]
yck9 = datck9[:,1]
xck11= -datck11[:,0]
yck11= datck11[:,1]
xck17= -datck17[:,0]
yck17= datck17[:,1]


xlf = -dat30[:,0]
ylf = dat30[:,1]

xhf  = -dat50[:,0]
yhf  = dat50[:,1]
sc=ax.scatter(xhf,yhf,marker='o',edgecolors='r',facecolors='none',s=17,alpha=1.,label='SU2')
sc=ax.scatter(xlf,ylf,marker='o',edgecolors='b',facecolors='none',s=17,alpha=1.,label='Correlation')
sc=ax.scatter(xck9,yck9,marker='o',c='magenta',s=17,alpha=1.,label='cokriging-9')
sc=ax.scatter(xck11,yck11,marker='o',c='purple',s=17,alpha=1.,label='cokriging-11')
sc=ax.scatter(xck17,yck17,marker='o',c='orange',s=17,alpha=1.,label='cokriging-17')

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)

ax.set_title('Pareto-front')
ax.set_xlabel('L/D',fontsize=14)
ax.set_ylabel('Mass (kg)',fontsize=14)
plt.legend()
plt.show()


quit()
thlhc   = [16,26,36,46,56]
th = np.array([[0.70111245, 0.0289625,  0.18313701],
[2.88390519, 0.02561265, 0.27699442],
[1.43166044, 0.01340619, 0.13444331],
[2.14877602, 0.05234007, 0.17499819],
[1.73083842, 0.02258635, 0.12119698]])
labels = ['Span','Taper','Sweep']

thhf = np.array([[2.0767774,  0.87499529, 0.14915089],
[2.4252076,  0.32054826, 0.19689324],
[2.11729458, 0.31610983, 0.08403438],
[6.42755286, 0.29458747, 0.11493633],
[6.68986794, 0.30150737, 0.10375832]])
thhflhc = [17, 22, 27, 32, 37]




fig,ax = plt.subplots()
ax.grid()
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.xaxis.grid(color='gray', linestyle='dashed')
cols = cm.brg(np.linspace(0,1,3))

for i in range(0,3):
    xi  = thlhc
    yi  = th[:,i]
    sc=ax.plot(xi,yi,marker='o',c=cols[i],linewidth=2,alpha=1.,label=labels[i])

#cax, _ = matplotlib.colorbar.make_axes(ax)
#cmap = matplotlib.cm.get_cmap('brg')
#normalize = matplotlib.colors.Normalize(vmin=10,vmax=50)
#cbar = matplotlib.colorbar.ColorbarBase(cax,cmap=cmap,norm=normalize)
#cbar.set_label('LHC size',rotation=270)
ax.set_title('Theta values')
ax.set_xlabel('LHC base size')
ax.set_ylabel('Theta')
plt.legend()
plt.show()

fig,ax = plt.subplots()
ax.grid()
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.xaxis.grid(color='gray', linestyle='dashed')
cols = cm.brg(np.linspace(0,1,3))

for i in range(0,3):
    xi  = thhflhc
    yi  = thhf[:,i]
    sc=ax.plot(xi,yi,marker='o',c=cols[i],linewidth=2,alpha=1.,label=labels[i])

#cax, _ = matplotlib.colorbar.make_axes(ax)
#cmap = matplotlib.cm.get_cmap('brg')
#normalize = matplotlib.colors.Normalize(vmin=10,vmax=50)
#cbar = matplotlib.colorbar.ColorbarBase(cax,cmap=cmap,norm=normalize)
#cbar.set_label('LHC size',rotation=270)
ax.set_title('Theta values')
ax.set_xlabel('LHC base size')
ax.set_ylabel('Theta')
plt.legend()
plt.show()





