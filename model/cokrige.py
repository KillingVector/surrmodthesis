# cokrige.py num 2 :(
# by A.A.Wachman 

import numpy as np
import math as m
import scipy
from scipy.optimize import minimize
from cokrige_matops import matrixops
from numpy.matlib import rand,zeros,ones,empty,eye
from pyKriging import kriging
from pyKriging import samplingplan
import random

import inspyred
from inspyred import ec

from matplotlib import pyplot as plt
import pylab
from mpl_toolkits.mplot3d import axes3d

import copy
from random import Random
from time import time




class cokriging(matrixops):
    def __init__(self, Xc, yc, Xe, ye,init=True):#init=False doesn't run processes, used with Load
        self.Xe = np.atleast_2d(Xe) 
        self.Xc = np.atleast_2d(Xc)
        print np.shape(self.Xe)
        print np.shape(self.Xc)
        #    will need to make yc, ye transpose/column vectors
        self.ye = np.atleast_2d(ye).T
        self.yc = np.atleast_2d(yc).T

        self.nc = Xc.shape[0]   # rows - num samps
        self.kc = Xc.shape[1]   # cols - num vars
        self.ne = Xe.shape[0]   # same shit but expensive
        self.ke = Xe.shape[1]   # 
        if init:
            self.normRange = []
            self.ynormRange = []
            self.normalizeData()

            
            print '-yc-'
            for i in range(0,len(self.yc)):
                print self.inversenormy(self.yc[i])
            print '-ye-'
            for i in range(0,len(self.ye)):
                print self.inversenormy(self.ye[i])

            #   this gives us the values for sigsqrc, muc, thetac and pc
            self.kcheap  = None#kriging(self.Xc, self.yc)
    #        self.kcheap.train()
            self.SigmaSqrc = None
            self.muc    = None
            self.thetac = np.ones(self.kc)
            self.p      = 2.
            self.pc     = 2.
            self.pd     = 2.
            self.rho    = 2. # rho regression parameter
            self.thetad = np.ones(self.kc)

            # training values
            self.thetamin = 1e-5 #* np.ones(self.kc)
            self.thetamax = 100 #* np.ones(self.kc)
            self.pmin = 1.5#0. #not gonna try handle noise atm
            self.pmax = 2.
            self.rhomin = 0.
            self.rhomax = 6.


            # final model values
            self.y          = None
            self.SigmaSqr   = None
            self.mu        = None
            self.theta      = None



            matrixops.__init__(self)
            self.updateDifferenceVector()

        print 'Co-kriging init complete' 


#####################################################################
#########################    GENERAL    #############################
#####################################################################


    #   Model update
    def updateModel(self):
        '''
        The function rebuilds the Psi matrix to reflect new data or a change in hyperparamters

        cokriging : also updates difference_vector()
        '''
        try:
            self.updateData()
            self.updatePsi()    
        except Exception as err:
            #pass
            # print Exception, err
            raise Exception("bad params")




#####################################################################
#########################   OPTIMISER   #############################
#####################################################################

    #   ===  GENERATOR  ===
    def generate_population(self, random, args):
        '''
        Generates an initial population for any global optimization that occurs in pyKriging
        :param random: A random seed
        :param args: Args from the optimizer, like population size
        :return chromosome: The new generation for our global optimizer to use
        '''
        size = args.get('num_inputs', None)
        bounder = args["_ec"].bounder
        chromosome = []
        for lo, hi in zip(bounder.lower_bound, bounder.upper_bound):
            chromosome.append(random.uniform(lo, hi))
#        print chromosome
#        quit()
        return chromosome  

    #   ===  OBJECTIVE  ===
    #   C   C   C   C   C   C
    def fittingObjectivec(self,candidates, args):
        '''
        The objective for a series of candidates from the hyperparameter global search.
        :param candidates: An array of candidate design vectors from the global optimizer
        :param args: args from the optimizer
        :return fitness: An array of evaluated NegLNLike values for the candidate population

                FOR CHEAP DATA
        '''
        fitness = []
        for entry in candidates:
            f=10000
            for i in range(self.kc):
                self.thetac[i] = entry[i]
            self.p     = entry[len(entry)-2]
            self.rho = entry[-1]
            try:
                self.updateModel()
                self.neglikelihoodc()
                f = self.NegLnLikec
            except Exception as e:
                f = 10000
            fitness.append(f)
        return fitness 
    #   LOCAL C
    def fittingObjective_localc(self,entry):
        '''
        :param entry: The same objective function as the global optimizer, but formatted for the local optimizer
        :return: The fitness of the surface at the hyperparameters specified in entry
        '''
        f=10000
        for i in range(self.kc):
            self.thetac[i] = entry[i]
        self.p     = entry[len(entry)-2]
        self.rho = entry[-1]
        try:
            self.updateModel()
            self.neglikelihoodc()
            f = self.NegLnLikec
        except Exception as e:
            # print 'Failure in NegLNLike, failing the run'
            # print Exception, e
            f = 10000
        return f


    #   D   D   D   D   D   D
    def fittingObjectived(self,candidates, args):
        '''
        The objective for a series of candidates from the hyperparameter global search.
        :param candidates: An array of candidate design vectors from the global optimizer
        :param args: args from the optimizer
        :return fitness: An array of evaluated NegLNLike values for the candidate population

                    FOR DIFFERENCE DATA
        '''
        fitness = []
        for entry in candidates:
            f=10000
            for i in range(self.ke):
                self.thetad[i] = entry[i]
            self.p     = entry[len(entry)-2]
            self.rho = entry[-1]
            try:
                self.updateModel()
                self.neglikelihoodd()
                f = self.NegLnLiked
            except Exception as e:
                f = 10000
            fitness.append(f)
        return fitness 
    #   LOCAL D
    def fittingObjective_locald(self,entry):
        '''
        :param entry: The same objective function as the global optimizer, but formatted for the local optimizer
        :return: The fitness of the surface at the hyperparameters specified in entry
        '''
        f=10000
        for i in range(self.ke):
            self.thetad[i] = entry[i]
        self.p     = entry[len(entry)-2]
        self.rho = entry[-1]
        try:
            self.updateModel()
            self.neglikelihoodd()
            f = self.NegLnLike
        except Exception as e:
            # print 'Failure in NegLNLike, failing the run'
            # print Exception, e
            f = 10000
        return f  

    #   ===  TERMINATOR  ===
    def no_improvement_termination(self, population, num_generations, num_evaluations, args):
        """Return True if the best fitness does not change for a number of generations of if the max number
        of evaluations is exceeded.

        .. Arguments:
           population -- the population of Individuals
           num_generations -- the number of elapsed generations
           num_evaluations -- the number of candidate solution evaluations
           args -- a dictionary of keyword arguments

        Optional keyword arguments in args:

        - *max_generations* -- the number of generations allowed for no change in fitness (default 10)

        """
        max_generations = args.setdefault('max_generations', 10)
        previous_best = args.setdefault('previous_best', None)
        max_evaluations = args.setdefault('max_evaluations', 30000)
        current_best = np.around(max(population).fitness, decimals=4)
        if previous_best is None or previous_best != current_best:
            args['previous_best'] = current_best
            args['generation_count'] = 0
            return False or (num_evaluations >= max_evaluations)
        else:
            if args['generation_count'] >= max_generations:
                return True
            else:
                args['generation_count'] += 1
                return False or (num_evaluations >= max_evaluations)



    #   === TRAINER ===
    def train_indiv(self, dataset = 'c', optimizer='pso'):
        '''
        The function trains the hyperparameters of the Kriging model.
        :param optimizer: Two optimizers are implemented, a Particle Swarm Optimizer or a GA

            dataset setting chooses 'cheap','c' for cheap data
                                    'diff', 'd' for diff data
                    if you type 'exp' or 'e' it also does 'diff'
        '''
        # check data type
        if dataset in ['cheap', 'c', 'kriging', 'krige']:
            obj     = self.fittingObjectivec
            obj_loc = self.fittingObjective_localc
            k       = self.kc
            theta   = self.thetac
            p       = self.p
            diff    = False
        elif dataset in ['diff', 'd', 'exp', 'e', 'cokriging', 'ck']:
            obj     = self.fittingObjectived
            obj_loc = self.fittingObjective_locald
            k       = self.kc
            theta   = self.thetad
            p       = self.p
            rho     = self.rho
            diff    = True




        # Establish the bounds for optimization for theta and p values
        lowerBound = [self.thetamin] * k + [self.pmin] + [self.rhomin]
        upperBound = [self.thetamax] * k + [self.pmax] + [self.rhomax]

        #Create a random seed for our optimizer to use
        rand = Random()
        rand.seed(int(time()))

        # If the optimizer option is PSO, run the PSO algorithm
        if optimizer == 'pso':
            ea = inspyred.swarm.PSO(Random())
            ea.terminator = self.no_improvement_termination
            ea.topology = inspyred.swarm.topologies.ring_topology
            # ea.observer = inspyred.ec.observers.stats_observer
            final_pop = ea.evolve(generator=self.generate_population,
                                  evaluator=obj,
                                  pop_size=300,
                                  maximize=False,
                                  bounder=ec.Bounder(lowerBound, upperBound),
                                  max_evaluations=30000,
                                  neighborhood_size=20,
                                  num_inputs=k)
            # Sort and print the best individual, who will be at index 0.
            final_pop.sort(reverse=True)

        # If not using a PSO search, run the GA
        elif optimizer == 'ga':
            ea = inspyred.ec.GA(Random())
            ea.terminator = self.no_improvement_termination
            final_pop = ea.evolve(generator=self.generate_population,
                                  evaluator=obj,
                                  pop_size=300,
                                  maximize=False,
                                  bounder=ec.Bounder(lowerBound, upperBound),
                                  max_evaluations=30000,
                                  num_elites=10,
                                  mutation_rate=.05)

        # This code updates the model with the hyperparameters found in the global search
        for entry in final_pop:
            newValues = entry.candidate
            preLOP = copy.deepcopy(newValues)
            locOP_bounds = []
            for i in range(k):
                locOP_bounds.append( [self.thetamin, self.thetamax] )

            locOP_bounds.append( [self.pmin , self.pmax] )
            locOP_bounds.append([self.rhomin,self.rhomax])

            # Let's quickly double check that we're at the optimal value by running a quick local optimizaiton
            lopResults = minimize(obj_loc, newValues, method='SLSQP', bounds=locOP_bounds, options={'disp': False})

            newValues = lopResults['x']

            # Finally, set our new theta and pl values and update the model again

            for i in range(k):
                theta[i] = newValues[i]
            p     = newValues[len(newValues)-2]

            if diff:
                rho = newValues[-1]
            
            try:
                self.updateModel()
            except:
                pass
            else:
                break


    #   ===  FULL  TRAINER  ===
    def co_trainer(self, optimiser='pso'):
        """
            This function runs both training schemes to get
        """
        print 'init rho : ' + str(self.rho)
        # First make sure our data is up-to-date
        self.updateData()
        print 'RIGHT BEFORE'
        self.updateModel()
        self.train_indiv(dataset = 'c', optimizer=optimiser)
        self.train_indiv(dataset = 'd', optimizer=optimiser)
        self.updatePsi()
        #   now we have thetac, thetad, pc and pd, also mus and sig2s
        print 'Theta C'
        print self.thetac
        print 'Theta D'
        print self.thetad
        print 'rho : ' + str(self.rho)
        print 'mu_d     : ' + str(self.mud)
        print 'Sig2d    : ' + str(self.SigmaSqrd)
        print 'mu_c     : ' + str(self.muc)
        print 'Sig2c    : ' + str(self.SigmaSqrc)


        print 'training complete'

    def train(self):
        """
            This function just does the whole training process
        """
        self.co_trainer()
        self.buildC()

        return


#####################################################################
#####################   PREDICTING SEC  #############################
#####################################################################

    def normX(self, X):
        '''
        :param X: An array of points (self.k long) in physical world units
        :return X: An array normed to our model range of [0,1] for each dimension
        '''

        X = copy.deepcopy(X)
        if type(X) is np.float64:
            # print self.normRange
            return np.array( (X - self.normRange[0][0]) / float(self.normRange[0][1] - self.normRange[0][0]) )
        else:
            for i in range(self.kc):
                X[i] = (X[i] - self.normRange[i][0]) / float(self.normRange[i][1] - self.normRange[i][0])
        return X

    def inversenormX(self, X):
        '''

        :param X: An array of points (self.k long) in normalized model units
        :return X : An array of real world units
        '''
        X = copy.deepcopy(X)
        for i in range(self.kc):
            X[i] = (X[i] * float(self.normRange[i][1] - self.normRange[i][0] )) + self.normRange[i][0]
        return X

    def normy(self, y):
        '''
        :param y: An array of observed values in real-world units
        :return y: A normalized array of model units in the range of [0,1]
        '''
#        print self.ynormRange
        yn0 = self.ynormRange[0]
        yn1 = self.ynormRange[1]
        return (y - yn0) / (yn1 - yn0)

    def inversenormy(self, y):
        '''
        :param y: A normalized array of model units in the range of [0,1]
        :return: An array of observed values in real-world units
        
        this works great normally for all positive ranges, but to accomodate 
        all neg ranges, we need to change this up a bit.(this may only be an issue because
        I am using -L/D, so the range is technically positive.

        '''

#        print 'ynorm range'
#        print self.ynormRange[1]
#        print self.ynormRange[0]

#           check if all negative ranges
#        neg = all(i < 0 for i in [self.ynormRange[0],self.ynormRange[1]])
#        neg = False

#        if neg:
#            y0 = -self.ynormRange[1]
#            y1 = -self.ynormRange[0]
#        elif not neg:
        y0 = self.ynormRange[0]
        y1 = self.ynormRange[1]


        return ((y * (y1 - y0)) + y0)


#        return (y * (self.ynormRange[1] - self.ynormRange[0])) + self.ynormRange[0]



    def normalizeData(self):    # from krige.py, modified for ck
        '''
        This function is called when the initial data in the model is set.
        We find the max and min of each dimension and norm that axis to a range of [0,1]
        '''
        for i in range(self.kc):
            self.normRange.append([min(self.Xc[:, i]), max(self.Xc[:, i])])

        # print self.X
        for i in range(self.nc):
            self.Xc[i] = self.normX(self.Xc[i])
        for i in range(self.ne):
            self.Xe[i] = self.normX(self.Xe[i])

        #   IMPORTANT: because I'm using -L/D I have to take 'abs' the ys
        ycmin   = min(self.yc)
        yemin   = min(self.ye)
        ycmax   = max(self.yc)
        yemax   = max(self.ye)
#        self.ynormRange.append(min([ycmin,yemin]))
#        self.ynormRange.append(max([ycmax,yemax]))
        self.ynormRange.append(yemin)
        self.ynormRange.append(yemax)

#        print self.ynormRange
        yhold = []
        for i in range(self.nc):
#            self.yc[i] = self.normy(self.yc[i])
            yhold.append(self.normy(self.yc[i]))
        self.yc  = np.array(yhold)
        yhold = []
        for i in range(self.ne):
            yhold.append(self.normy(self.ye[i]))
        self.ye  = np.array(yhold)




    def predict_normalized(self,var):
        Xe  = self.Xe
        Xc  = self.Xc
        ye  = self.ye   
        yc  = self.yc
        ne  = self.ne
        nc  = self.nc
#        print Xe
#        print ye
#        print Xc
#        print yc

        thetad  = self.thetad
        thetac  = self.thetac
        p       = self.p#2.
        rho     = self.rho

        one     = np.ones((nc+ne,1))
        y       = self.y

        cc      = np.ones((nc,1))
        for i in range(0,nc):
            distcc  = np.power(np.abs(Xc[i,:] - var), p)
            cc[i,0] = rho * self.SigmaSqrc * np.exp(-np.sum(np.multiply(thetac,distcc)))

        cd      = np.ones((ne,1))
        for i in range(0,ne):
            distcd  = np.power(np.abs(Xe[i,:] - var), p)
            cd[i,0] = rho**2 * self.SigmaSqrc * np.exp(-np.sum(np.multiply(thetac,distcd))) + self.SigmaSqrd * np.exp(-np.sum(np.multiply(thetad,distcd)))

        c   = np.concatenate((cc,cd), axis=0)

        f1  = (y - one.dot(self.mu[0][0]))
        f2  = np.linalg.pinv(self.C)
        f3  = f2.dot(f1)

        f   = self.mu + c.T.dot(f3)

        fuck = f
#        print self.ynormRange
        return fuck # however many fucks I give


    def predict(self, X):
        '''
        This function returns the prediction of the model at the real world coordinates of X
        :param X: Design variable to evaluate
        :return: Returns the 'real world' predicted value
        '''
        X = copy.deepcopy(X)
        X = self.normX(X)
#        print 'fin'
#        print self.ynormRange
        prediction  = self.inversenormy(self.predict_normalized(X))

        return prediction[0][0]





    def save_ck_model(self,filename):
        import pickle
        Xe  = self.Xe
        Xc  = self.Xc
        ye  = self.ye   
        yc  = self.yc
        ne  = self.ne
        nc  = self.nc

        thd  = self.thetad
        thc  = self.thetac
        p       = self.p#2.
        rho     = self.rho

        y       = self.y

        xnorm   = self.normRange
        ynorm   = self.ynormRange

        sig2c   = self.SigmaSqrc
        sig2d   = self.SigmaSqrd
        muc     = self.muc  
        mud     = self.mud
        mu      = self.mu
        C       = self.C

        with open(filename+'.pkl','w') as f:
            pickle.dump([Xe,Xc,ye,yc,ne,nc,thd,thc,p,rho,y,xnorm,ynorm,sig2c,sig2d,muc,mud,mu,C],f)
            # this should save all the good bits that we need to predict as a .pkl file

        return

    def load_ck_model(self,filename):
        import pickle
        

        with open(filename+'.pkl') as f:
            Xe,Xc,ye,yc,ne,nc,thd,thc,p,rho,y,xnorm,ynorm,sig2c,sig2d,muc,mud,mu,C = pickle.load(f)

        self.Xe  = Xe
        self.Xc  = Xc
        self.ye  = ye   
        self.yc  = yc
        self.ne  = ne
        self.nc  = nc

        self.thetad = thd
        self.thetac = thc
        self.p      = p#2.
        self.rho    = rho

        self.y      = y

        self.normRange    = xnorm
        self.ynormRange   = ynorm

        self.SigmaSqrc    = sig2c
        self.SigmaSqrd    = sig2d
        self.muc          = muc
        self.mud          = mud  
        self.mu           = mu
        self.C            = C



        return


    

def fc(X):
    return np.power(X[:,0],2) + X[:,0] + np.power(X[:,1],2) + X[:,1] + np.power(X[:,2],2) + X[:,2] + np.power(X[:,3],2) + X[:,3] + np.power(X[:,4],2) + X[:,4]
def fe(X):
    return np.power(X[:,0], 2) + np.power(X[:,1], 2) + np.power(X[:,2],2)+ np.power(X[:,3],2) + np.power(X[:,4],2) 

if __name__=='__main__':

    Xc  = np.array([[7.5, 3.75], [19.5, -4.25], [16.5, -2.25], [58.5, -1.25], [55.5, -3.25], [49.5, 2.25], [28.5, -0.25], [1.5, 2.75], [0.0, -5.0], [0.0, 5.0], [60.0, -5.0], [60.0, 5.0], [7.5, -1.25], [52.5, 1.25], [22.5, 3.75], [37.5, -3.75]])
    yc  = np.array([[-17.733460106073434, 120.81460542413934], [-18.463904113774177, 121.4457348907803], [-18.222789283084214, 121.23958286244964], [-24.5691836123388, 132.57827847832112], [-23.960815547053055, 130.62165397397393], [-22.788736588026843, 127.63626835013986], [-19.41857655978084, 122.3882173709343], [-17.606995591579548, 120.67694140921077], [-17.600627042340786, 120.65981164363428], [-17.60062704267269, 120.65981164363428], [-24.89297883556556, 133.84707430576168], [-24.892978835005565, 133.84707430576168], [-17.733460105785536, 120.81460542413934], [-23.366847693627204, 128.99964626833977], [-18.7520681679817, 121.7736825513534], [-20.716833914947284, 124.18039568938633]])

    yc  = yc[:,0]

    Xe  = np.array([[7.5, -1.25], [52.5, 1.25], [22.5, 3.75], [37.5, -3.75]])

    ye  = np.array([[-17.733462569298666, 120.81460542413934], [-23.51226992639444, 128.99964626833977], [-18.752087151783677, 121.7736825513534], [-22.922963852071955, 124.18039568938633]])
    ye  = ye[:,0]

#    sp = samplingplan(5)
#    Xc = sp.rlh(10)
#    Xe = np.array( random.sample(Xc, 6) )
#    #    Xe = sp.rlh(6)     # deal with this case later


#    yc = fc(Xc)
#    ye = fe(Xe)

#    f1 = fc(np.array([Xe[1,:]]))
#    f2 = fe(np.array([Xe[1,:]]))
#    f3 = fc(np.array([Xe[2,:]]))
#    f4 = fe(np.array([Xe[2,:]]))
#    f5 = fc(np.array([Xe[3,:]]))
#    f6 = fe(np.array([Xe[3,:]]))
#    f7 = fc(np.array([Xe[4,:]]))
#    f8 = fe(np.array([Xe[4,:]]))

#    print 'prior\ncheap'+str(Xc)
#    print 'expen'+str(Xe)

    ck = cokriging(Xc, yc, Xe, ye)
#    quit()
#    ck.updatePsi()
#    ck.neglikelihoodc()
#    ck.neglikelihoodd()
#    quit()
#    ck.train_indiv(dataset='c')
#    ck.train_indiv(dataset='d')



#    ck.co_trainer()
#    quit()
#    ck.buildC()

#    ck.save_ck_model('fucktits')
    ck.load_ck_model('fucktits')


#    print 'Xe : ' + str(Xe)
#    print 'ye : ' + str(ye)
    print '======== result'
    print 'yc ' + str(-20.716833914947284)
    print 'ye ' + str(-22.922963852071955)
    point = [37.5, -3.75] # -25.89194581, 77.39316109
    v = np.array(point)
    print 'predict'
    print ck.predict(v)
    print 'norm predict'
    print ck.predict_normalized(v)
#    print fc(v)
#    print fe(v)

#    print ck.cokrigingpredictor(Xe[1,:])
#    print f1
#    print f2
#    print ck.cokrigingpredictor(Xe[2,:])
#    print f3
#    print f4
#    print ck.cokrigingpredictor(Xe[3,:])
#    print f5
#    print f6
#    print ck.cokrigingpredictor(Xe[4,:])
#    print f7
#    print f8


#    yd = []
#    for i in range(0,10):
#        yd.append(ck.cokrigingpredictor(Xc[i,:]))

#    kc = kriging(Xc, yc)
#    kc.plot()
#    kl = kriging(Xc, yd)
#    kl.plot()
#    ke = kriging(Xe, ye)
#    ke.plot()







#    Xc  = np.array([[ 5.75, 47.,   -1.5,   2.5,  -3.5,   3.5 ],\
#     [ 3.85, 11.,   -3.5,   0.5,   3.5,   1.5 ],\
#     [13.35, -7.,    2.5,   3.5,  -0.5,   0.5 ],\
#     [11.45, 35.,    0.5,   4.5,   4.5,  -2.5 ],\
#     [19.05, 29.,    1.5,  -4.5,   0.5,  -0.5 ],\
#     [15.25, -1.,   -4.5,  -3.5,  -1.5,   2.5 ],\
#     [ 9.55,  5.,   -0.5,  -1.5,   1.5,  -4.5 ],\
#     [ 7.65, 41.,    3.5,  -2.5,   2.5,   4.5 ],\
#     [ 1.95, 17.,    4.5,  -0.5,  -2.5,  -1.5 ],\
#     [17.15, 23.,   -2.5,   1.5,  -4.5,  -3.5 ]])

#    ally    = np.array([[-14.59333506,  66.51344181],\
#     [-11.72859694,  64.36265929],\
#     [-32.24489567,  88.30290174],\
#     [-30.74693477,  82.87115012],\
#     [-38.62322819, 104.66688081],\
#     [-34.47040908,  93.76846908],\
#     [-25.80428418,  77.39316109],\
#     [-21.00818414,  71.96536036],\
#     [-11.81816674,  64.36744794],\
#     [-37.18276976,  99.18047181]])
#    yc  = ally[:,0]

#    Xe  = np.array([[ 9.55,  5.,   -0.5,  -1.5,   1.5,  -4.5 ],\
#     [ 3.85, 11.,   -3.5,   0.5,   3.5,   1.5 ]])

#    yeall  = np.array([[-25.89194581,  77.39316109],\
#     [ -9.27444075,  64.36265929]])

#    ye  = yeall[:,0]










#    Xc = np.array(\
#    [[0.89473684, 0.47368421, 0.63157895, 0.47368421, 0.15789474, 0.        ,  0.        , 0.52631579, 0.36842105, 0.68421053],\
#     [  8.42105263e-01,   0.00000000e+00,   7.89473684e-01,   1.05263158e-01,    5.78947368e-01,   9.47368421e-01,   7.89473684e-01,   4.73684211e-01,    5.26315789e-01,   9.47368421e-01],\
#     [  1.05263158e-01,   3.68421053e-01,   4.21052632e-01,   5.26315789e-02,    2.63157895e-01,   6.84210526e-01,   8.94736842e-01,   1.57894737e-01,    2.63157895e-01,   5.26315789e-02],\
#     [  2.10526316e-01,   3.15789474e-01,   5.78947368e-01,   1.00000000e+00,    6.84210526e-01,   4.73684211e-01,   8.42105263e-01,   1.00000000e+00,    5.26315789e-02,   7.36842105e-01],\
#     [  6.31578947e-01,   2.63157895e-01,   3.68421053e-01,   7.36842105e-01,    7.89473684e-01,   7.36842105e-01,   5.78947368e-01,   4.21052632e-01,    1.00000000e+00,   7.89473684e-01],\
#     [  4.73684211e-01,   9.47368421e-01,   8.42105263e-01,   8.94736842e-01,    5.26315789e-01,   8.94736842e-01,   2.10526316e-01,   2.10526316e-01,    5.78947368e-01,   0.00000000e+00],\
#     [1.        , 0.68421053, 0.89473684, 0.68421053, 0.42105263, 0.57894737,  0.68421053, 0.31578947, 0.        , 0.84210526],\
#     [0.78947368, 0.42105263, 0.15789474, 0.31578947, 0.31578947, 0.42105263,  0.26315789, 0.10526316, 0.73684211, 0.89473684],\
#     [0.94736842, 0.84210526, 0.73684211, 0.        , 0.36842105, 0.52631579,  0.36842105, 0.36842105, 0.84210526, 0.31578947],\
#     [0.26315789, 0.73684211, 1.        , 0.57894737, 0.89473684, 0.26315789,  0.42105263, 0.84210526, 0.10526316, 0.21052632],\
#     [0.52631579, 1.        , 0.31578947, 0.63157895, 0.10526316, 0.31578947,  0.52631579, 0.        , 0.31578947, 0.36842105],\
#     [  6.84210526e-01,   5.26315789e-02,   4.73684211e-01,   1.57894737e-01,    9.47368421e-01,   2.10526316e-01,   3.15789474e-01,   9.47368421e-01,    9.47368421e-01,   1.05263158e-01],\
#     [  4.21052632e-01,   2.10526316e-01,   2.63157895e-01,   4.21052632e-01,    8.42105263e-01,   8.42105263e-01,   5.26315789e-02,   2.63157895e-01,    1.57894737e-01,   1.57894737e-01],\
#     [  3.68421053e-01,   6.31578947e-01,   1.05263158e-01,   5.26315789e-01,    5.26315789e-02,   7.89473684e-01,   9.47368421e-01,   6.84210526e-01,    8.94736842e-01,   4.73684211e-01],\
#     [  5.26315789e-02,   5.78947368e-01,   9.47368421e-01,   7.89473684e-01,    2.10526316e-01,   3.68421053e-01,   1.00000000e+00,   5.78947368e-01,    4.73684211e-01,   6.31578947e-01],\
#     [  7.36842105e-01,   7.89473684e-01,   5.26315789e-02,   2.63157895e-01,    7.36842105e-01,   1.05263158e-01,   7.36842105e-01,   7.89473684e-01,    6.31578947e-01,   4.21052632e-01],\
#     [0.57894737, 0.10526316, 0.21052632, 0.84210526, 0.47368421, 0.05263158,  0.15789474, 0.89473684, 0.21052632, 0.26315789],\
#     [  3.15789474e-01,   5.26315789e-01,   0.00000000e+00,   9.47368421e-01,    1.00000000e+00,   6.31578947e-01,   6.31578947e-01,   5.26315789e-02,    4.21052632e-01,   5.26315789e-01],\
#     [  0.00000000e+00,   1.57894737e-01,   5.26315789e-01,   2.10526316e-01,    0.00000000e+00,   1.00000000e+00,   4.73684211e-01,   7.36842105e-01,    6.84210526e-01,   5.78947368e-01],\
#     [0.15789474, 0.89473684, 0.68421053, 0.36842105, 0.63157895, 0.15789474,  0.10526316, 0.63157895, 0.78947368, 1.        ]])

#    yc = np.array([ 0.19876765,  0.11998783,  0.9624564,   0.96178546,  0.24319634,  0.49556195,  0.35225994,  0.         , 0.18351977,  0.84804796,  0.41615025,  0.21851555,  0.46134129,  0.55453634,  1.        ,  0.04395008,  0.26864421,  0.61613168,  0.98601731,  0.99816077])

#    Xe = np.array(\
#    [[0.89473684, 0.47368421, 0.63157895, 0.47368421, 0.15789474, 0.        ,  0.        , 0.52631579, 0.36842105, 0.68421053],\
#     [1.        , 0.68421053, 0.89473684, 0.68421053, 0.42105263, 0.57894737,  0.68421053, 0.31578947, 0.        , 0.84210526],\
#     [0.78947368, 0.42105263, 0.15789474, 0.31578947, 0.31578947, 0.42105263,  0.26315789, 0.10526316, 0.73684211, 0.89473684],\
#     [0.94736842, 0.84210526, 0.73684211, 0.        , 0.36842105, 0.52631579,  0.36842105, 0.36842105, 0.84210526, 0.31578947],\
#     [0.26315789, 0.73684211, 1.        , 0.57894737, 0.89473684, 0.26315789,  0.42105263, 0.84210526, 0.10526316, 0.21052632],\
#     [0.52631579, 1.        , 0.31578947, 0.63157895, 0.10526316, 0.31578947,  0.52631579, 0.        , 0.31578947, 0.36842105],\
#     [0.57894737, 0.10526316, 0.21052632, 0.84210526, 0.47368421, 0.05263158,  0.15789474, 0.89473684, 0.21052632, 0.26315789],\
#     [0.15789474, 0.89473684, 0.68421053, 0.36842105, 0.63157895, 0.15789474,  0.10526316, 0.63157895, 0.78947368, 1.        ]])

#    ye = np.array([ 0.14243009,  0.1543546 ,  0.        ,  0.19363917,  0.823362  ,  0.43969499,  0.28396734,  1.        ])
