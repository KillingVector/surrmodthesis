# test inspyred

# generator
# evaluator
# solver choice
# display


import pyKriging
import inspyred
import pylab as plt
import numpy as np
import SUAVE
import time, copy, inspect, random, itertools, matplotlib
import csv, os

from random import Random
from time import time

from pyKriging.krige import kriging  
from pyKriging.samplingplan import samplingplan

from SUAVE.Core import Data,Units
from SUAVE.Optimization import Nexus

from inspyred import ec
from inspyred.ec import emo
from inspyred.ec import selectors
from inspyred import swarm

from Surrogate_Nexus import Nexus
from surrogate import Surrogate_Data

from pyKriging import saveModel, loadModel
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from inspyred import ec
import itertools





class inspyred_wrapper(Data):


    def __init__(self, nexus, obj = 'krig'):
        optprob         = nexus.optimization_problem
        self.inputs     = optprob.inputs
        self.bounds     = self.make_bounds()
        self.lower_bound= self.bounds[1]
        self.upper_bound= self.bounds[0]
        self.bounder    = ec.Bounder(self.lower_bound, self.upper_bound)

        surr            = nexus.surrogate_data
        # if the models are not empty, make kriging_model.predict(inps) the funct
        if obj.lower() in ['k','krig','krige','kriging']:
            if not surr.model0 == None:
                self.function_1     = surr.model0.predict
            else:
                self.function_1     = None
            if not surr.model1 == None:
                self.function_2     = surr.model1.predict
            else:
                self.function_2     = None
        elif obj.lower() in ['ck','cokrig','co-krige','co-kriging','co kriging','cokriging','cok']:
            if not surr.modelck0 == None:
                self.function_1     = surr.modelck0.predict
            else:
                self.function_1     = None
            if not surr.model0 == None:
                self.function_2     = surr.model0.predict
            elif not surr.model1 == None:
                self.function_2     = surr.model1.predict
            else:
                self.function_2     = None
        elif obj.lower() in ['mix','other','borked','):']:
            if not surr.modelck0 == None:
                self.function_1     = surr.modelck0.predict
            else:
                self.function_1     = None
            if not surr.model1  == None:
                self.function_2     = surr.model1.predict
            else:
                self.function_2     = None
        # not gonna worry about 3 atm (:
#        print self.function_1
#        print self.function_2


    def __defaults__(self):

#        self.bounds = None
        self.names  = []
        # optimizer
        self.opt        = 'nsga2'
        self.pop_size   = 30
        self.method     = 'min'     # max or min need????
        self.generations= 100
    
        # counters
        self.genct      = 0
        self.evalct     = 0


        # final settings
        self.final_pop  = None
        self.ea_archive = None

        self.storecands = []
        self.storegens  = []

    
    # Used in __init__
    def make_bounds(self):

        inputs      = self.inputs
        names       = inputs[:,0]       # names
        bounds      = inputs[:,2]       # bounds [l,u]
        scale       = inputs[:,3]       # scaling
        units       = inputs[:,-1]*1.0
        inputs[:,-1]= units

        num_var = np.shape(names)[0]
        
        # get upper and lower bounds
        for i in range(0,num_var):
            if i == 0:
                ub = [bounds[i][1]]
                lb = [bounds[i][0]]
            else:
                ub.append(bounds[i][1])
                lb.append(bounds[i][0])
            self.names.append(names[i])
        return [ub, lb]


    def check_penalty(self, X):
        names   = self.names
        [ub, lb]  = self.bounds
        failed  = False

        # inside bounds check
#        failed = False
#        for i in range(0,len(X)):
#            if X[i] > ub[i]:
#                return True
#            elif X[i] < lb[i]:
#                return True

        for i in range(0,len(X)):
            if X[i] > ub[i]:
                X[i] = ub[i] - 0.01*ub[i]
            elif X[i] < lb[i]:
                X[i] = lb[i] + 0.01*lb[i]
#                return failed   # exit funct
        rcpnames   = []
        rcpvals    = [] 
        pslnames   = []
        pslvals    = []
        # reasonable values check
        for i in range(0,len(names)):
            name    = names[i]
            Xval    = X[i]
            if name[0:3] == 'rcp':
                rcpnames.append(name)
                rcpvals.append(Xval)
            if name[0:3] == 'psl':
                pslnames.append(name)
                pslvals.append(Xval)

        if len(rcpnames) > 1:   
            for i in range(1,len(rcpnames)):
                if rcpvals[i-1] < rcpvals[i]:
                    failed = True
                    return failed

        if len(pslnames) > 1:
            for i in range(1,len(rcpnames)):
                if pslvals[i-1] > pslvals[i]:
                    failed = True
                    return failed

        

        return failed

        


    def insp_generator(self, random, args):
#        print 'gen'
        self.genct = self.genct + 1
        [ub, lb]    = self.bounds
        gen         = []
        for i in range(0,len(ub)):
            gen.append(np.random.uniform(ub[i],lb[i]))
        X = gen
        # constrain the generated values
        for i in range(0,len(X)):
            if X[i] > ub[i]:
                X[i] = ub[i] - 0.01*ub[i]
            elif X[i] < lb[i]:
                X[i] = lb[i] + 0.01*lb[i]
        gen = X
#        print gen
        return gen

    def insp_bounder(self,candidates,args):


        return


    def insp_evaluator(self, candidates, args):
        #   check constraint bounds before evaluating
        [ub,lb]     = self.bounds
        self.storecands.append(candidates)
        self.evalct = self.evalct + 1
        fitness = []
        #   if failed constraint, penalize the candidate
        for X in candidates:
            failed = self.check_penalty(X)
#            failed = False

            f1      = self.function_1(X)
            f2      = self.function_2(X)


            if f1 <-26 or f1 >= 0:# f1 > 0 for neg lift/drag, 
                # f1 < 0 for drag/lift
                # 60 l/d is only going to happen with ridiculous glider, but is limit
#                print 'is failed?'
                # initial optimisations, -60, for this case we know -27 is too big
                failed = True

            if f2 <= 0 or f2 > 500:
#                print 'is failed2'
                failed = True


            if failed:  # if failed, penalise both functions
#                quit()
                f1  = f1 + 200      # order of magnitude larger than expected lift/drag
                f2  = f2 + 1000      # half an order of magnitude larger than expected mass
            else:
                fitness.append(emo.Pareto([f1, f2]))

#        for X in candidates:   # test case
#            f1 = X[0]**2 + sum([X[i]**2 for i in range(1,len(X))])
#            f2 = (X[0]-1)**2 + sum([X[i]**2 for i in range(1,len(X))])
#            fitness.append(emo.Pareto([f1, f2]))
        self.storegens.append(fitness)
        return fitness


    def evolve(self):
        prng            = Random()
        prng.seed(time())

        if self.opt in ['nsga2','nsgaii','NSGA2','NSGAII','NSGAii']:
            ea          = inspyred.ec.emo.NSGA2(prng)
            ea.variator = [inspyred.ec.variators.blend_crossover,
                           inspyred.ec.variators.gaussian_mutation]
            ea.terminator=inspyred.ec.terminators.generation_termination
            self.final_pop= ea.evolve(generator=self.insp_generator, 
                            evaluator=self.insp_evaluator, 
                            pop_size=self.pop_size,
#                            bounder=self.bounder,
                            maximize=False,
                            max_generations=self.generations)
            self.ea_archive=ea.archive

#        elif self.opt in ['PSO','pso','particle swarm','particle_swarm']:
#            ea          = inspyred.swarm.PSO(prng)
#            ea.terminator=inspyred.ec.terminators.evaluation_termination
#            ea.topology = inspyred.swarm.topologies.ring_topology

#            self.final_pop = ea.evolve(generator=self.insp_generator,
#                              evaluator=self.insp_evaluator, 
#                              pop_size=self.pop_size,
#                              bounder=problem.bounder,
#                              maximize=problem.maximize,
#                              max_evaluations=30000, 
#                              neighborhood_size=5)

        elif self.opt in ['pareto archived','paes','PAES']:
            ea          = inspyred.ec.emo.PAES(prng)
            ea.terminator=inspyred.ec.terminators.evaluation_termination
            self.final_pop= ea.evolve(generator=self.insp_generator, 
                              evaluator=self.insp_evaluator, 
#                              bounder=self.bounder,
                              maximize=False,
                              max_evaluations=10000,
                              max_archive_size=100,
                              num_grid_divisions=4)
            self.ea_archive=ea.archive

        else:
            print 'Only PAES, NSGA2 supported at the moment'

            
        return 

    def show_results(self,title='Optimisation results'):
        final_arc = self.ea_archive
        print('Best Solutions: \n')
        i = 0
        for f in final_arc:
            print(f)

        import pylab
        x = []
        y = []
        for f in final_arc:
            x.append(f.fitness[0])
            y.append(f.fitness[1])
        x = -np.array(x)
        y = np.array(y)
        fig = plt.figure()
        ax = fig.gca()
        ax.grid()
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.xaxis.grid(color='gray', linestyle='dashed')
        plt.scatter(x, y, color='b')
        plt.title(title)
        plt.xlabel('L/D')
        plt.ylabel('Mass (kg)')
#        ax.set_xticks(np.arange(0, 1, 0.1))
#        ax.set_yticks(np.arange(0, 1., 0.1))
        plt.show()

        return

    def show_gen_results(self,title='Generation evolution'):
        import matplotlib.cm as cm
        final_arc = self.ea_archive
        gens=[]
        div = 100
        lengen  = len(self.storegens)        
        num     = int(lengen / div)
        print num
        print lengen
        i = 0
        print self.storegens
        while i < len(self.storegens):
            gens.append(self.storegens[i])
            i = i + num
        
        x = []
        y = []
        for f in final_arc:
            x.append(f.fitness[0])
            y.append(f.fitness[1])
        x = -np.array(x)
        y = np.array(y)

#        fig = plt.figure()
#        ax  = plt.axes()
#        ax = fig.add_subplot(111)
        fig,ax = plt.subplots()
        ax.grid()
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.xaxis.grid(color='gray', linestyle='dashed')
        cols = cm.viridis(np.linspace(0,1,div+1))
        alpha = .2
        for i in range(0,len(gens)):
#            print i
            xi = -np.array(gens[i])[:,0]
            yi = np.array(gens[i])[:,1]
            sc=ax.scatter(xi,yi,marker='o',c=cols[i],s=7,alpha=0.5)
#            ax.scatter(xi,yi,marker='*')
            alpha = alpha + 0.8/(lengen/num)
#            print alpha
        ax.scatter(x, y, c=cols[len(gens)-1],marker='o',s=8)
        cax, _ = matplotlib.colorbar.make_axes(ax)
        cmap = matplotlib.cm.get_cmap('viridis')
        normalize = matplotlib.colors.Normalize(vmin=0,vmax=self.generations)
        cbar = matplotlib.colorbar.ColorbarBase(cax,cmap=cmap,norm=normalize)
        cbar.set_label('Generation',rotation=270)
        ax.set_title(title)
        ax.set_xlabel('L/D')
        ax.set_ylabel('Mass (kg)')
        plt.show()



        


    def get_results(self,quant = 1):
        print 'counts : gen = ' + str(self.genct) + '  eval = '+str(self.evalct)
        final_arc   = self.ea_archive
        ins         = []
        outs        = []
        for f in final_arc:
#            print f.fitness.values
            ins.append(f.candidate) # list
            outs.append(f.fitness.values)  # .values (string) else: class: inspyred.ec.emo.Pareto
        if quant > 0:
            a = []
            b = []
            for i in range(0,quant):
                a.append(ins[quant])
                b.append(ins[quant])
            return a, b
        else:
            return ins, outs

    def save_results(self, name = 'insp_save'):
        ins, outs   = self.get_results(quant=-1)
        saveModel([self.storecands,self.storegens],name+'cands_generations.pkl')
        # os.path.dirname(__file__) +
        with open( name + '.csv', 'w+b') as savefile:
            wr      = csv.writer(savefile, quoting=csv.QUOTE_ALL)
            for i in range(len(ins)):
                row = ins[i]
                row.append(outs[i][0])
                row.append(outs[i][1])
                wr.writerow(row)
            print 'write completed to : ' + name + '.csv'

        return
            





if __name__ == '__main__':
    
    iw = inspyred_wrapper()
#    for test case
#        comment lines 40-54, 110-116
#        uncomment lines 117-120
    iw.bounds       = [[2.,2.] , [1.,1.]]

    iw.evolve()
#    iw.show_results()
    iw.show_gen_results()
    ins, outs = iw.get_results(quant=-1)
    iw.save_results()
#    print outs
#   appears to work :")





    

