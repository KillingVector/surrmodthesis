import numpy as np
import scipy
import copy
from matplotlib import pyplot as plt
import pylab

from mpl_toolkits.mplot3d import axes3d
from time import time
from inspyred import ec
import math as m


print 'PLOTTER'

def kplot(kmod=None, labels=False, show=True, xlabel='x',ylabel='y',zlabel='z'):
    '''
    This function plots 2D and 3D models
    :param labels:
    :param show: If True, the plots are displayed at the end of this call. If False, plt.show() should be called outside this function
    :return:
    '''
    if kmod.k == 3:
        print 'KMOD3'
#        import mayavi.mlab as mlab

#        predictFig = mlab.figure(figure='predict')
        fig1 = plt.figure()
        # errorFig = mlab.figure(figure='error')
        if kmod.testfunction:
            truthFig = mlab.figure(figure='test')
        dx = 1
        pts = 25#j
        X, Y, Z = np.mgrid[0:dx:pts, 0:dx:pts, 0:dx:pts]
        scalars = np.zeros(X.shape)
        errscalars = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k1 in range(X.shape[2]):
                    # errscalars[i][j][k1] = kmod.predicterr_normalized([X[i][j][k1], Y[i][j][k1], Z[i][j][k1]])
                    scalars[i][j][k1] = kmod.predict_normalized([X[i][j][k1], Y[i][j][k1], Z[i][j][k1]])

        if kmod.testfunction:
            tfscalars = np.zeros(X.shape)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    for k1 in range(X.shape[2]):
                        tfplot = tfscalars[i][j][k1] = kmod.testfunction([X[i][j][k1], Y[i][j][k1], Z[i][j][k1]])
            plot = mlab.contour3d(tfscalars, contours=15, transparent=True, figure=truthFig)
            plot.compute_normals = False

        # obj = mlab.contour3d(scalars, contours=10, transparent=True)
        plot = mlab.contour3d(scalars, contours=15, transparent=True, figure=predictFig)
        plot.compute_normals = False
        # errplt = mlab.contour3d(errscalars, contours=15, transparent=True, figure=errorFig)
        # errplt.compute_normals = False
        mlab.xlabel(xlabel)
        mlab.ylabel(ylabel)
        mlab.zlabel(zlabel)
        if show:
            mlab.show()

    if kmod.k==2:
        print 'KMOD2'
        fig = pylab.figure(figsize=(8,6))
        samplePoints = list(zip(*kmod.X))
        # Create a set of data to plot
        plotgrid = 61
        x = np.linspace(kmod.normRange[0][0], kmod.normRange[0][1], num=plotgrid)
        y = np.linspace(kmod.normRange[1][0], kmod.normRange[1][1], num=plotgrid)

        # x = np.linspace(0, 1, num=plotgrid)
        # y = np.linspace(0, 1, num=plotgrid)
        X, Y = np.meshgrid(x, y)

        # Predict based on the optimized results

        zs = np.array([kmod.predict([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)
        # Z = (Z*(kmod.ynormRange[1]-kmod.ynormRange[0]))+kmod.ynormRange[0]

        #Calculate errors
        zse = np.array([kmod.predict_var([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Ze = zse.reshape(X.shape)

        spx = (kmod.X[:,0] * (kmod.normRange[0][1] - kmod.normRange[0][0])) + kmod.normRange[0][0]
        spy = (kmod.X[:,1] * (kmod.normRange[1][1] - kmod.normRange[1][0])) + kmod.normRange[1][0]
        contour_levels = 25

        ax = fig.add_subplot(222)
        CS = pylab.contourf(X,Y,Ze, contour_levels)
        pylab.colorbar()
        pylab.plot(spx, spy,'ow')

        ax = fig.add_subplot(221)
        if kmod.testfunction:
            # Setup the truth function
            zt = kmod.testfunction( np.array(list(zip(np.ravel(X), np.ravel(Y)))) )
            ZT = zt.reshape(X.shape)
            CS = pylab.contour(X,Y,ZT,contour_levels ,colors='k',zorder=2)


        # contour_levels = np.linspace(min(zt), max(zt),50)
        if kmod.testfunction:
            contour_levels = CS.levels
            delta = np.abs(contour_levels[0]-contour_levels[1])
            contour_levels = np.insert(contour_levels, 0, contour_levels[0]-delta)
            contour_levels = np.append(contour_levels, contour_levels[-1]+delta)

        CS = plt.contourf(X,Y,Z,contour_levels,zorder=1)
        pylab.plot(spx, spy,'ow', zorder=3)
        pylab.colorbar()

        ax = fig.add_subplot(212, projection='3d')
        # fig = plt.gcf()
        #ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=3, cstride=3, alpha=0.4)
        if kmod.testfunction:
            ax.plot_wireframe(X, Y, ZT, rstride=3, cstride=3)
        if show:
            pylab.show()

def kplotdubs(kmod=None, kmod2=None, labels=False, show=True, xlabel='x',ylabel='y',zlabel='z'):
    '''
    This function plots 2D and 3D models
    :param labels:
    :param show: If True, the plots are displayed at the end of this call. If False, plt.show() should be called outside this function
    :return:
    '''
    if kmod.k == 3:
        import mayavi.mlab as mlab

        predictFig = mlab.figure(figure='predict')
        # errorFig = mlab.figure(figure='error')
        if kmod.testfunction:
            truthFig = mlab.figure(figure='test')
        dx = 1
        pts = 25j
        X, Y, Z = np.mgrid[0:dx:pts, 0:dx:pts, 0:dx:pts]
        scalars = np.zeros(X.shape)
        scalars2= np.zeros(X.shape)
        errscalars = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k1 in range(X.shape[2]):
                    # errscalars[i][j][k1] = kmod.predicterr_normalized([X[i][j][k1], Y[i][j][k1], Z[i][j][k1]])
                    scalars[i][j][k1] = kmod.predict_normalized([X[i][j][k1], Y[i][j][k1], Z[i][j][k1]])    
                    scalars2[i][j][k1] = kmod2.predict_normalized([X[i][j][k1], Y[i][j][k1], Z[i][j][k1]])    
                    errscalars[i][j][k1] = scalars[i][j][k1] - scalars2[i][j][k1]

        if kmod.testfunction:
            tfscalars = np.zeros(X.shape)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    for k1 in range(X.shape[2]):
                        tfplot = tfscalars[i][j][k1] = kmod.testfunction([X[i][j][k1], Y[i][j][k1], Z[i][j][k1]])
            plot = mlab.contour3d(tfscalars, contours=15, transparent=True, figure=truthFig)
            plot.compute_normals = False
    
        scalars = scalars - scalars2
        print errscalars
        quit()
        # obj = mlab.contour3d(scalars, contours=10, transparent=True)
#        mlab.contour3d(scalars2, contours=15, transparent=True, figure=predictFig)
        plot = mlab.contour3d(errscalars, contours=15, transparent=True, figure=predictFig)
        
        plot.compute_normals = False
        # errplt = mlab.contour3d(errscalars, contours=15, transparent=True, figure=errorFig)
        # errplt.compute_normals = False
        mlab.xlabel(xlabel)
        mlab.ylabel(ylabel)
        mlab.zlabel(zlabel)
        if show:
            mlab.show()

    if kmod.k==2:

        fig = pylab.figure(figsize=(8,6))
        samplePoints = list(zip(*kmod.X))
        # Create a set of data to plot
        plotgrid = 61
        x = np.linspace(kmod.normRange[0][0], kmod.normRange[0][1], num=plotgrid)
        y = np.linspace(kmod.normRange[1][0], kmod.normRange[1][1], num=plotgrid)

        # x = np.linspace(0, 1, num=plotgrid)
        # y = np.linspace(0, 1, num=plotgrid)
        X, Y = np.meshgrid(x, y)

        # Predict based on the optimized results

        zs = np.array([kmod.predict([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)
        # Z = (Z*(kmod.ynormRange[1]-kmod.ynormRange[0]))+kmod.ynormRange[0]

        #Calculate errors
        zse = np.array([kmod.predict_var([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Ze = zse.reshape(X.shape)

        spx = (kmod.X[:,0] * (kmod.normRange[0][1] - kmod.normRange[0][0])) + kmod.normRange[0][0]
        spy = (kmod.X[:,1] * (kmod.normRange[1][1] - kmod.normRange[1][0])) + kmod.normRange[1][0]
        contour_levels = 25

        ax = fig.add_subplot(222)
        CS = pylab.contourf(X,Y,Ze, contour_levels)
        pylab.colorbar()
        pylab.plot(spx, spy,'ow')

        ax = fig.add_subplot(221)
        if kmod.testfunction:
            # Setup the truth function
            zt = kmod.testfunction( np.array(list(zip(np.ravel(X), np.ravel(Y)))) )
            ZT = zt.reshape(X.shape)
            CS = pylab.contour(X,Y,ZT,contour_levels ,colors='k',zorder=2)


        # contour_levels = np.linspace(min(zt), max(zt),50)
        if kmod.testfunction:
            contour_levels = CS.levels
            delta = np.abs(contour_levels[0]-contour_levels[1])
            contour_levels = np.insert(contour_levels, 0, contour_levels[0]-delta)
            contour_levels = np.append(contour_levels, contour_levels[-1]+delta)

        CS = plt.contourf(X,Y,Z,contour_levels,zorder=1)
        pylab.plot(spx, spy,'ow', zorder=3)
        pylab.colorbar()

        ax = fig.add_subplot(212, projection='3d')
        # fig = plt.gcf()
        #ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=3, cstride=3, alpha=0.4)
        if kmod.testfunction:
            ax.plot_wireframe(X, Y, ZT, rstride=3, cstride=3)
        if show:
            pylab.show()




def ckplot(kmod=None, kmod2=None, labels=False, show=True, xlabel='x',ylabel='y',zlabel='z'):
    '''
    This function plots 2D and 3D models
    :param labels:
    :param show: If True, the plots are displayed at the end of this call. If False, plt.show() should be called outside this function
    :return:
    '''
    if kmod.k == 3:
        import mayavi.mlab as mlab

        predictFig = mlab.figure(figure='predict')
        # errorFig = mlab.figure(figure='error')
        if kmod.testfunction:
            truthFig = mlab.figure(figure='test')
        dx = 1
        pts = 25j
        X, Y, Z = np.mgrid[0:dx:pts, 0:dx:pts, 0:dx:pts]
        scalars = np.zeros(X.shape)
        scalars2= np.zeros(X.shape)
        errscalars = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k1 in range(X.shape[2]):
                    # errscalars[i][j][k1] = kmod.predicterr_normalized([X[i][j][k1], Y[i][j][k1], Z[i][j][k1]])
                    scalars[i][j][k1] = kmod.predict_normalized([X[i][j][k1], Y[i][j][k1], Z[i][j][k1]])    
                    scalars2[i][j][k1] = kmod2.predict([X[i][j][k1], Y[i][j][k1], Z[i][j][k1]])    

        if kmod.testfunction:
            tfscalars = np.zeros(X.shape)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    for k1 in range(X.shape[2]):
                        tfplot = tfscalars[i][j][k1] = kmod.testfunction([X[i][j][k1], Y[i][j][k1], Z[i][j][k1]])
            plot = mlab.contour3d(tfscalars, contours=15, transparent=True, figure=truthFig)
            plot.compute_normals = False

        # obj = mlab.contour3d(scalars, contours=10, transparent=True)
        mlab.contour3d(scalars2, contours=15, transparent=True, figure=predictFig)
        plot = mlab.contour3d(scalars, contours=15, transparent=True, figure=predictFig)
        
        plot.compute_normals = False
        # errplt = mlab.contour3d(errscalars, contours=15, transparent=True, figure=errorFig)
        # errplt.compute_normals = False
        mlab.xlabel(xlabel)
        mlab.ylabel(ylabel)
        mlab.zlabel(zlabel)
        if show:
            mlab.show()

    if kmod.k==2:

        fig = pylab.figure(figsize=(8,6))
        samplePoints = list(zip(*kmod.X))
        # Create a set of data to plot
        plotgrid = 61
        x = np.linspace(kmod.normRange[0][0], kmod.normRange[0][1], num=plotgrid)
        y = np.linspace(kmod.normRange[1][0], kmod.normRange[1][1], num=plotgrid)

        # x = np.linspace(0, 1, num=plotgrid)
        # y = np.linspace(0, 1, num=plotgrid)
        X, Y = np.meshgrid(x, y)

        # Predict based on the optimized results

        zs = np.array([kmod.predict([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)
        # Z = (Z*(kmod.ynormRange[1]-kmod.ynormRange[0]))+kmod.ynormRange[0]

        #Calculate errors
        zse = np.array([kmod.predict_var([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Ze = zse.reshape(X.shape)

        spx = (kmod.X[:,0] * (kmod.normRange[0][1] - kmod.normRange[0][0])) + kmod.normRange[0][0]
        spy = (kmod.X[:,1] * (kmod.normRange[1][1] - kmod.normRange[1][0])) + kmod.normRange[1][0]
        contour_levels = 25

        ax = fig.add_subplot(222)
        CS = pylab.contourf(X,Y,Ze, contour_levels)
        pylab.colorbar()
        pylab.plot(spx, spy,'ow')

        ax = fig.add_subplot(221)
        if kmod.testfunction:
            # Setup the truth function
            zt = kmod.testfunction( np.array(list(zip(np.ravel(X), np.ravel(Y)))) )
            ZT = zt.reshape(X.shape)
            CS = pylab.contour(X,Y,ZT,contour_levels ,colors='k',zorder=2)


        # contour_levels = np.linspace(min(zt), max(zt),50)
        if kmod.testfunction:
            contour_levels = CS.levels
            delta = np.abs(contour_levels[0]-contour_levels[1])
            contour_levels = np.insert(contour_levels, 0, contour_levels[0]-delta)
            contour_levels = np.append(contour_levels, contour_levels[-1]+delta)

        CS = plt.contourf(X,Y,Z,contour_levels,zorder=1)
        pylab.plot(spx, spy,'ow', zorder=3)
        pylab.colorbar()

        ax = fig.add_subplot(212, projection='3d')
        # fig = plt.gcf()
        #ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=3, cstride=3, alpha=0.4)
        if kmod.testfunction:
            ax.plot_wireframe(X, Y, ZT, rstride=3, cstride=3)
        if show:
            pylab.show()







if __name__ == '__main__':



















