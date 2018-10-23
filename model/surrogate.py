# imports general
import numpy as np
import datetime, time, csv, inspect, random, itertools

import SUAVE
from SUAVE.Core import Data

import pyKriging
from pyKriging import saveModel, loadModel
from pyKriging.krige import kriging  
from pyKriging.samplingplan import samplingplan

from MakeVehicle import generate_vehicle

from cokrige import cokriging as ckrig

#from plot_krige



class Surrogate_Data(Data):


    def __defaults__(self):
        #   Setup values
        self.op                 = Data()
        self.op.names           = []
        self.fidelity_level     = -1

        # base aircraft - get these values from time runs
        self.lf_res        = None
        self.hf_res        = None

        self.vec                = Data()
        self.vec.init_vec       = None
        self.vec.make_vec       = None      # store each it of make_vehicle
        self.vec.material       = None      # main vehicle material
        self.vec.aerofoil       = None      # put aerofoil data here
        self.vec.payload        = None      # payload data

        #   Sample points and LHC settings
        self.sample_plan    = Data()        # full set of initial values, uses pyKriging sampleplan
        self.sample_plan.lhc_type   = 'optimal'  #rlh, optim
        self.sample_plan.olhc_pop   = 30
        self.sample_plan.olhc_iter  = 30
        self.sample_plan.size       = 20   # if 0 or less than 2*input_dim, uses in built setting
        #   Final data set
        self.sample_plan.time       = 15. * 60. # seconds
        self.sample_plan.lhc= None      # use create_sample()    
        self.sample_plan.lhc_lf = None      
        self.sample_plan.lhc_mf = None
        self.sample_plan.lhc_hf = None
        self.sample_plan.exp_freq=0.2
        self.sample_plan.names  = None
        # set the following using evaluate_of()
        self.X              = None      # successful X
        self.X_e            = None
        self.X_ck           = None
        self.y              = None      # successful evaluations
        self.y_e            = None
        self.y_ck           = None
        self.y_fail         = None      # failed or invalid evaluations
        self.g              = None
        self.g_e            = None
        self.g_ck           = None

        #   Model and optimizer
        self.model0         = None # at least initially this is for kriging, use pyKriging
        self.model1         = None
        self.modelck0       = None # storage for cokriging models
        self.modelckplot       = None
        self.optimizer      = None # need this to work with general inputs and any opt package
        self.objective_funct= None
        #   Optimizer settings
#        self.opt_



    
    def info(self):
        """
            This function prints info of this class
        """
        # will print all available options in __defaults__
        # will also print available functions
        pass


    # create sample
    def ck_create_sample(self,nexus):
        #   which models do we have?

        if self.op.names == []:
            if hasattr(self.op,'lf'):
                self.op.names.append('lf')
            if hasattr(self.op,'mf'):
                self.op.names.append('mf')
            if hasattr(self.op,'hf'):
                self.op.names.append('hf')
#        print self.op.names
        if len(self.op.names) <= 1:
            print 'One or fewer models present.'
            print 'Please add another modeling fidelity level before continuing.'
            quit()
        #   set low and high
        if 'hf' in self.op.names:
            high  = self.op.hf
            if 'mf' in self.op.names:
                low    = self.op.mf
                config  = 'mfhf'
                print '=== CONFIG : ' + config
            else:
                low     = self.op.lf
                config  = 'lfhf'
                print '=== CONFIG : ' + config
        else:
            high    = self.op.mf
            low     = self.op.lf
            config  = 'lfmf'    # not using this atm
            print '=== CONFIG : ' + config

        var_num    = len(low.optimization_problem.inputs[:,0]) # cols = num vars
        # get run times for one each
#        lt1         = datetime.datetime.now()
        lowres      = None#low.objective()
#        lowtime     = (datetime.datetime.now() - lt1).total_seconds()
#        ht1         = datetime.datetime.now()
        highres     = None#high.objective()
#        hightime    = (datetime.datetime.now() - ht1).total_seconds()

        if config == 'lfmf':
            hightime = 0.7 # AVL time elapsed approx 0.7s
        elif config == 'mfhf':
            lowtime = 0.7
        lowtime = 1.
        hightime = 100.

#        self.lf_res        = lowres
#        self.hf_res        = highres
        #   FIX ME
        #   get ratio of run times
        ratiotime   = lowtime/hightime
        time_all    = self.sample_plan.time # allowed run time
        num_run1    = time_all/lowtime      # number of low fid runs
        groups      = 1 + ratiotime       # ratio of low to high TIMES
        low_sample  = num_run1 / groups     # low sample is one portion
        num_run2    = num_run1 - low_sample # get remaining number of low fid times
        high_sample = (num_run2 * lowtime) / (1/ratiotime) # get equiv high fid number
        print 'sample sizes:'

        print ratiotime
        print low_sample
        print high_sample

        # time is essential
        low_sample = int(np.floor(low_sample))
        high_sample= int(np.floor(high_sample))
        print 'time for single low fid run : ' + str(lowtime) + ' s'
        print 'time for single high fid run : ' + str(hightime) + ' s'

        ratiorun = 1/ratiotime
        

#        quit()
        if config == 'lfmf':
            if low_sample > 50 * var_num:
                low_sample = 50 * var_num
            if high_sample > 10 * var_num:
                high_sample = 10 * var_num
        else:
            if low_sample > 50 * var_num:
                low_sample = 50 * var_num
            if high_sample > 20 * var_num:
                high_sample = 20 * var_num

        # make corners
        corn    = []
        for i in itertools.product([0,1],repeat=var_num*1):
            i = list(i)
            corn.append(i)
        corn = np.array(corn)
        corn = self.scale_points(corn,nexus)

        print config
        print low_sample
        print high_sample
        if low_sample >30 and low_sample <40:
            high_sample = 11
        elif low_sample >= 50 and low_sample < 75:
            high_sample = 13
        elif low_sample > 75:
            if config == 'lfmf':
                high_sample = 35
            else:
                high_sample = 15

#        print np.shape(corn)
#        quit()
        # create base lhc sample
        low_sample  = low_sample #- high_sample #- np.floor(np.shape(corn)[0]/2)  # adjust size
        print low_sample
        print '==='
        low_lhc     = self.lhc_sample(nexus,dimension=['low',low_sample])
        high_lhc    = self.lhc_sample(nexus,dimension=['high',high_sample])
#        data = np.genfromtxt('fullsu2rc1.7520km.csv',delimiter=',')
#        high_lhc = data[:,0:3]
#        print high_lhc


        print 'Adding corners... (size will vary with number of variables)'
        low_lhc     = np.concatenate((low_lhc,corn),axis=0)
        print 'LHC starting dimension : ' + str(np.shape(low_lhc))
        print 'Adding LHC of high fidelity of size : '+str(np.shape(high_lhc))

        gauge   = 0.01 * max(np.absolute(high_lhc.flatten()))
        numlist = 0
        for hrow in high_lhc:
            end = False
            for i in range(np.shape(low_lhc)[0]-1,-1,-1):
                lrow    = low_lhc[i,:]
                dist    = np.absolute(np.linalg.norm(hrow-lrow))
                if dist < gauge:
                    numlist = numlist+1
                    low_lhc = np.delete(low_lhc,i,0)  #ith row (axis-0)

        low_lhc     = np.concatenate((low_lhc,high_lhc), axis=0)
        print 'Number deleted = ' +str( numlist       )
        
        self.sample_plan.lhc        = low_lhc
        if config == 'lfmf':
            print 'lfmf'
            self.sample_plan.lhc_lf = low_lhc
            self.sample_plan.lhc_mf = high_lhc
        elif config == 'lfhf':
            self.sample_plan.lhc_lf = low_lhc
            self.sample_plan.lhc_hf = high_lhc
        elif config == 'mfhf':
            self.sample_plan.lhc_mf = low_lhc
            self.sample_plan.lhc_hf = high_lhc

        print 'LHC final dimension : ' + str(np.shape(self.sample_plan.lhc))
        print np.shape(high_lhc)

#        quit()
        return


    def create_sample(self,nexus):
        """
            This creates the single fidelity run sample
        """


        self.sample_plan.lhc    = self.lhc_sample(nexus)
        self.sample_plan.lhc_lf = self.sample_plan.lhc
        print type(self.sample_plan.lhc)
        # add corners
        size    = np.shape(self.sample_plan.lhc)[1] # cols = num vars
        corn    = []
        for i in itertools.product([0,1],repeat=size*1):
            i = list(i)
            corn.append(i)
        corn = np.array(corn)
        corn = self.scale_points(corn,nexus)
        
        print 'LHC starting dimension : ' + str(np.shape(self.sample_plan.lhc))

        print 'Adding corners... (size will vary with number of variables)'
        
        self.sample_plan.lhc = np.concatenate((self.sample_plan.lhc,corn),axis=0)

        print 'LHC final dimension : ' + str(np.shape(self.sample_plan.lhc))

#        quit()
        return





        
    def lhc_sample(self,nexus,dimension=['',-1]):
        """
            this function creates and scales the sample based on 
            the input upper and lower bounds.
        """

        fid         = dimension[0]
        size        = dimension[1]

        if size == -1:
            size = self.sample_plan.size

        inputs      = nexus.optimization_problem.inputs

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

        # this should always perform, but in case is already ndarray:
        if isinstance(ub, np.ndarray) == False:
            ub      = np.array(ub)
            lb      = np.array(lb)
#            print 'did lb ub'

        # make sample - latin hypercube: random (fast), optimal (slow)
        sampleplan      = samplingplan(num_var)

        # Get lhc sample size
        # ==== FOR GENERAL RUN, SINGLE FID
        if fid == '':
            size = self.sample_plan.size

            if size == 0 or size == -1 or None: # choose defaults
                print   'LHC gen: defaults chosen'
                if num_var <= 10:
                    sample_size = 10
                else:
                    sample_size = 2 * num_var
            elif not size == 0:   # controlled user choice
                print   'LHC gen: user choice'
                if size < num_var:
                    sample_size = 2 *num_var   # prevent silly choices
                    print 'sample size too small, using minimum (number of variables or 10)'
                else:
                    sample_size = size# - num_var **2

        #   ==== FOR LOW FIDELITY OF RUN
        elif fid == 'low':
            if size == 0 or size == -1: # choose defaults
                print   'LHC gen: defaults chosen'
                if num_var <= 10:
                    sample_size = 10
                else:
                    sample_size = 10 * num_var
            elif not size == 0:   # controlled user choice
                print   'LHC gen: user choice'
                if size < 10 * num_var:
                    sample_size = 10 * num_var   # prevent silly choices
                    print 'sample size too small, using minimum (number of variables or 10)'
                elif size > 100 - num_var**2:
                    sample_size = 100 - num_var**2
                else:
                    sample_size = size
        #   ==== FOR HIGH FIDELITY OF RUN
        elif fid == 'high':
            if size == 0 or size == -1: # choose defaults
                print   'LHC gen: defaults chosen'
                sample_size = 4 * num_var
            elif not size == 0:   # controlled user choice
                print   'LHC gen: user choice'
                if size < 4 * num_var:
                    sample_size = 4 * num_var   # prevent silly choices
                    print 'sample size too small, using minimum (number of variables or 10)'
                elif size > 30:
                    sample_size = 30
                else:
                    sample_size = size

        print sample_size
        # check if optimal or random
        try:
            if self.sample_plan.lhc_type.lower() in ['olh','optimal','opt','best','sehr gut','o'] and size <= 50:
                print 'Optimal Latin Hypercube Sample'
                sample          = sampleplan.optimallhc(sample_size,population=20, iterations=20, generation=False)
            elif self.sample_plan.lhc_type in ['rlh','rand','random','r','Rando Calrissian'] or size > 50:
                print 'Random Latin Hypercube Sample'
                sample          = sampleplan.rlh(sample_size)
                # -- !!! worth checking how much time impact this actually has
        except:
            use_your_words = 'Incorrect designation for latin hypercube sample type.\nPlease choose from:\n Optimal, eg: \'olh\',\'optimal\',\'o\'\n Random, eg: \'rlh\',\'rand\',\'random\',\'r\''
            raise Exception(use_your_words)
            exit()  # make them do it again

        # scale lhc sample to inputs
        scale           = ub - lb               # find range
        scaled_inputs   = np.multiply(scale,sample) + lb   # correct for lb
        #   SAVE
#        self.sample_plan.lhc = scaled_inputs


        return scaled_inputs





    #   need to scale points whenever inserted into objfunct
    def scale_points(self,incoming_points,nexus):
        """
            incoming_points     [some vector of n variable points]

        """

        inputs      = nexus.optimization_problem.inputs

        names       = inputs[:,0]       # names
        bounds      = inputs[:,2]       # bounds [l,u]
        scale       = inputs[:,3]       # scaling
        units       = inputs[:,-1]*1.0
        inputs[:,-1]= units

        num_var = np.shape(names)[0]
        
        # oh look we are repeating code wow
        for i in range(0,num_var):
            if i == 0:
                ub = [bounds[i][1]]
                lb = [bounds[i][0]]
            else:
                ub.append(bounds[i][1])
                lb.append(bounds[i][0])
        if isinstance(ub, np.ndarray) == False:
            ub      = np.array(ub)
            lb      = np.array(lb)

        scale           = ub - lb               # find range
        scaled_inputs   = np.multiply(scale,incoming_points) + lb   # correct for lb

        return scaled_inputs
        


#    def objective_function(self,X):
    def objective_function(self,inp_i,nexus,obj_choice=-1):

        """
            This function evaluates a single set of variables.
            These functions should describe the vehicle, analysis level and mission.
            At some point I'll make a way to specify which fidelity level is required.

            The get_results function can be modified to produce different results.  This will be used to create two surrogate models that can be used for multiobjective analysis.
        """


#        nexus           = self.nexus
        optprob         = nexus.optimization_problem    # i need this :/
        # iterate through surrogate model sample items imported into nexus items
        for i in range(0,len(inp_i)):
            optprob.inputs[:,1][i]  = inp_i[i]
#            self.make_vec           = generate_vehicle(self.vec)

        try:
            obj             = nexus.objective()*optprob.objective[0][1]
    #        constraints_val = nexus.all_constraints().tolist()

            if obj_choice == -1:
                f = obj
            else:
                f = obj[obj_choice]

    #        g = constraints_val
            print 'objective ===: D/L ==== ' + str(f)

            fail = 0
            if obj[0] >= 0 or obj[1] <= 0:     # -L/D and -mass make no sense 
            # change sign direction on obj[0] if looking for maximizing -lift
                fail = 1
            elif np.isnan(obj[0]) or np.isnan(obj[1]):
                fail = 1
################################################################################################
            elif obj[0] < -40.: #   this is a constraint: observed that high fid
                                #   never gets higher than 25
                fail = 1
################################################################################################

        except Exception as err:
            print err
            print 'Failing objective'
            fail = 1
            f = [-10000, -10000]

        return f, fail





    #   === get results for 
    def evaluate_of(self,nexus,config=''):
        print "Evaluating objective function"
        """
            Evaluates objective function for each lhc row of the model variables.
            
            Assumptions:
                Function acting as the objective has been set.  Throws error if not.

            Source:
                N/A

            Inputs:
                None

            Outputs:
                None, updates Surrogate_Data structure

        """
        if config == '':
            scaled_inputs   = self.sample_plan.lhc
        elif config in ['','low','l']:
            if hasattr(self.op,'hf') and self.fidelity_level == 2:
                scaled_inputs   = self.sample_plan.lhc_mf   # if higher fid exists, use avl
            elif self.fidelity_level == 1:
                scaled_inputs   = self.sample_plan.lhc_lf
        elif config in ['high','h','hi']:
            if hasattr(self.op,'hf') and self.fidelity_level == 2:
                scaled_inputs   = self.sample_plan.lhc_hf
            elif self.fidelity_level == 1:
                scaled_inputs   = self.sample_plan.lhc_mf
            
        y       = []
#        g       = []
        X       = []
        y_fail  = []

#        with open('./avlkriging_set'+str(datetime.datetime.now())+'.csv','w+b') as file:
#            wr=csv.writer(file)
        for j in range(np.shape(scaled_inputs)[0]):
            X_j         = scaled_inputs[j,:]
            print X_j
            f ,fail     = self.objective_function(X_j,nexus)
#                f,gi,fail= self.objective_function(X_j,nexus)
#                print type(f)
#                print type(X_j)


            if fail == 0:
                y.append(f.tolist())
#                    g.append(gi)
                X.append(X_j.tolist())

#                    row = []
#                    for item in X_j:
#                        row.append(item)
#                    for item2 in f:
#                        row.append(item2)
#                    wr.writerow(row)
            else: # fail = 1
                y_fail.append(f)    # bunch of nans

        
        if config in ['','low','l']:
            self.X              = np.array(X)
            self.y              = np.array(y)
            self.y_fail         = np.array(y_fail)
        elif config in ['high','h','hi']:
            self.Xe             = np.array(X)
            self.ye             = np.array(y)
            self.y_fail         = np.array(y_fail)


        return 




    #   KRIGING - SINGLE FID
    def single_fid_kriging(self,nexus, improve=False):
        print "Generating kriging model ..."
        """
            single_fidelity_kriging
            obj_choice      =   choice of objective to generate model, 0 for first/default
                                subsequent functions for others.
            
            At the moment this is only single fidelity kriging.
            Will look into allowing both co-kriging and heirarchical kriging
            Option  = 'heirarchical' or 'hk', and 'cokriging', 'co-kriging' 
                    or 'ck'
        """
        if not isinstance(self.y, np.ndarray):
            y       = np.array(self.y)
        else:
            y       = self.y

        if len(y[0,:]) > 1:
            two_mod = True
        else:
            two_mod = False


        #   check input types
        if not isinstance(self.X, np.ndarray):
            X       = np.array(self.X)
        else:
            X       = self.X

        if two_mod:
            y0      = y[:,0]
            y1      = y[:,1]
        else:
            y0      = y
#        print '\n ================= \n'
#        print X
#        print y

        of          = self.objective_function

        tk1 = time.time()

        k0  = kriging(X, y0)
        k0.train()
        print 'First training: model 0'
        if two_mod:
            k1  = kriging(X, y1)
            k1.train()
            print 'First training: model 1'
            
        tk2 = time.time()
#        print k.X
#        print k.y
        print 'Kriging model initial setup time : ' + str(tk2 - tk1) + ' sec'



        # We need to check before we update (getting an uninvertible numpy matrix :/ )
        # Method 1: remove a point and compare its original predicted value with k_red.predict(x)
#        check   = self.kriging_model_check(k)
        # Method 2: make a new point
        checkpt1= self.scale_points(k0.X.mean(0),nexus)
        checkpt2= self.scale_points(k0.X[0:np.shape(X)[0]/2,:].mean(0),nexus)

        if improve:
            ei              = np.minimum(k0.expimp(checkpt1),k0.expimp(checkpt2))
        else:
           ei = 1.e-10 
        counter         = 0
        gain_small      = False
        print 'precheck : ' + str(ei)
        while ei >= 1.e-4 and not gain_small and counter <= 5:# and improve:# guarantees a few updates
            print 'Kriging infill iteration ' + str(counter)
            newpoints   = k0.infill(1)   # this is being a wild prick >:(         
#            print 'are these reasonable' +str(newpoints)
            ei          = k0.expimp(newpoints)
            print 'Expected Improvement : EI(x) = ' + str(ei)
            if ei < 1e-4:
                gain_small = True
            else:
                for point in newpoints:
                    value   = of(point,nexus,-1)
#                    print "\n\n value ===== " + str(value)
                    if len(value[0]) > 1:
                        obj_0   = value[0][0]
                        obj_1   = value[0][1]
                    else:
                        obj_0   = value[0]
                    if value[1]==0: 
                        k0.addPoint(point, obj_0)
                        k0.train()
                        if len(value[0]) >1 and two_mod:
                            k1.addPoint(point,obj_1)
                            k1.train()
            counter = counter + 1
        tk3 = time.time()
        print 'Exit criteria: gain_small = ' + str(gain_small) + '  counter = '+ str(counter)+'/10'
        print 'Kriging model improvement time : ' + str(tk3 - tk2) + ' sec'
        print 'Total time : ' + str(tk3 - tk1) + ' sec'
        
        
        self.model0 = k0
        print 'model 0 added'
        if two_mod:
            self.model1 = k1
            print 'model 1 added'



        return #k



########################################################################
########################      co - kriging      ########################
################################ METHOD ################################


    #   FOR FULL RUN
    def hybrid_cokriging(self,nexus,filename='cokriging'):
        
        #   which models do we have?
        if self.op.names == []:
            if hasattr(self.op,'lf'):
                self.op.names.append('lf')
            if hasattr(self.op,'mf'):
                self.op.names.append('mf')
            if hasattr(self.op,'hf'):
                self.op.names.append('hf')
#        print self.op.names
        if len(self.op.names) <= 1:
            print 'One or fewer models present.'
            print 'Please add another modeling fidelity level before continuing.'
            quit()
        #   set low and high
        if 'hf' in self.op.names:
            high  = self.op.hf
            if 'mf' in self.op.names:
                low    = self.op.mf
                config  = 'mfhf'
            else:
                low     = self.op.lf
                config  = 'hflf'
        else:
            high    = self.op.mf
            low     = self.op.lf
            config  = 'lfmf'    # not using this atm

        config = 'lfhf'
#        self.ck_create_sample(nexus)
        data1 = np.genfromtxt('/home/ashaiden/Documents/surrmodthesis/model/rawresults/kriging/kriging_XY/k0-spantpsw-X-y.csv',delimiter=',')
        data2 = np.genfromtxt('/home/ashaiden/Documents/surrmodthesis/model/rawresults/kriging/kriging_XY/k2-sptpsw-X-y-9x3.csv',delimiter=',')

        self.X = data1[:,0:3]
        self.sample_plan.lhc = self.X
        self.sample_plan.lhc_mf = self.X
        self.y = data1[:,3:5]

        self.Xe = data2[:,0:3]
        self.sample_plan.lhc_hf = self.Xe

        self.ye = data2[:,3:5]
        

        

#        self.evaluate_of(low,config='low')
#        data = np.genfromtxt('cheaps2018-10-15 12:00:42.781182.csv',delimiter=',')
#        self.y = data[:,3:5]
#        self.X = data[:,0:3]
        thetime = str(datetime.datetime.now())

#        # some saving routines
#        with open('./cheaps'+thetime+'.csv','w+b') as filec:
#            wrc=csv.writer(filec)
#            for i in range(0,np.shape(self.X)[0]):
#                row = self.X[i,:].tolist()
#                row.append(self.y[i,:])
#                wrc.writerow(row)

#        self.evaluate_of(high,config='high')
        ###################################################################
#        data = np.genfromtxt('fullsu2rc1.7520km.csv',delimiter=',')
#        self.ye = data[:,3:5]
#        self.Xe = data[:,0:3]
        ###################################################################

#        with open('./exp'+thetime+'.csv','w+b') as filee:
#            wre=csv.writer(filee)
#            for i in range(0,np.shape(self.Xe)[0]):
#                row = self.Xe[i,:].tolist()
#                row.append(self.ye[i,:])
#                wre.writerow(row)



        #   Check if proper type
        if not isinstance(self.y, np.ndarray):
            yc      = np.array(self.y)
        else:
            yc      = self.y
        if not isinstance(self.ye, np.ndarray):
            ye      = np.array(self.ye)
        else:
            ye      = self.ye
        if not isinstance(self.X, np.ndarray):
            X_cheap = np.array(self.X)
        else:
            X_cheap = self.X
        if not isinstance(self.Xe, np.ndarray):
            X_exp   = np.array(self.Xe)
        else:
            X_exp  = self.Xe

        print 'assuming mass model saved'
#        print 'Building and training mass model'
#        masskrig        = kriging(X_cheap,yc[:,1])
#        masskrig.train()
#        self.model0     = masskrig
#            #   SAVE MODEL
#        saveModel(self.model0,filename+'mass.pkl')

        #   this co-kriges the L/D data
        print X_cheap
        print 'Building and training model 1'
        ck0     = ckrig(X_cheap, yc[:,0], X_exp, ye[:,0])
        ck0.co_trainer()
        ck0.buildC()
        print 'co-kriging model 1 complete\n'
        

        self.modelck0   = ck0
            #   SAVE MODEL
        self.modelck0.save_ck_model(filename+str(np.shape(X_exp)))
        '''
        Xck = self.scale_points(self.model0.X,nexus)
            #   make l/d model for comparison
        kld = kriging(X_cheap,yc[:,0])
        kld.train()
        self.model1 = kld

        with open('./'+config+'cokriging'+thetime+'.csv','w+b') as file1:
            wr = csv.writer(file1)
            for i in range(0,np.shape(self.X)[0]):
                inp = Xck[i,:]
                row = Xck[i,:].tolist()
                pred= self.modelck0.predict(inp).tolist()
                pred= np.array(pred)
                row.append(pred)
                try:
                    pred2=np.array(self.model1.predict(inp).tolist())
                    row.append(pred2)
                except:
                    'do nothing'
                wr.writerow(row)

        '''
        return thetime  # means get_ck_plot can be called outside this funct


###################################         LOADING AND PLOTTING FROM FILES

    def defaultck(self):
        """
            this will be called before loading a cokriging model
        """
        Xc  = np.array([[7.5, 3.75], [19.5, -4.25], [16.5, -2.25], [37.5, -3.75]])
        yc  = np.array([[-17.733460106073434, 120.81460542413934], [-18.463904113774177, 121.4457348907803], [-18.222789283084214, 121.23958286244964], [-20.716833914947284, 124.18039568938633]])
        yc = yc[:,0]
        Xe  = np.array([[7.5, -1.25],[37.5, -3.75]])
        ye  = np.array([[-17.733462569298666, 120.81460542413934], [-22.922963852071955, 124.18039568938633]])
        ye  = ye[:,0]
        ck = ckrig(Xc,yc,Xe,ye,init=False)
        self.modelck0 = ck

        return


    def load_ck(self,filename):
        """
            First loads default ck model and then overwrites it
        """
    
        self.defaultck()

        ck  = self.modelck0

        self.modelck0.load_ck_model(filename)
        try:
            self.model0    = loadModel(filename+'mass.pkl')
        except Exception as err:
            print err
            print 'You have not saved a relevant mass kriging model'
        return


    def save_ck(self,filename):
        """
            Saves current cokriging and kriging model
        """
        ck = self.modelck0
        ck.save_ck_model(filename)

        try:
            k   = self.model0
            saveModel(k,filename+'mass.pkl')
        except Exception as err:
            print err
            print 'You have not created a relevant mass kriging model'
        return

        






########################################################################
#######################       cokrig plot       ########################
########################################################################


    def get_plot(self,nexus,model=None,model1=None,zlabel='',mapname='viridis'):
        """
            Somehow I can do many other things, but I cannot plot things. 
            I am very stupid
            update: i can plot the things. Upgraded from potato to yam.
        """
        from mpl_toolkits import mplot3d
#        matplotlib inline
        import matplotlib.pyplot as plt

        inputs      = nexus.optimization_problem.inputs

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

        # this should always perform, but in case is already ndarray:
        if isinstance(ub, np.ndarray) == False:
            ub      = np.array(ub)
            lb      = np.array(lb)

        size = 50

        lx1  = np.linspace(lb[0],ub[0],size)
        lx2  = np.linspace(lb[1],ub[1],size)
        
        x1, x2 = np.meshgrid(lx1,lx2)
        
        gen = np.zeros(np.shape(x1))

        namelist = []
        for item in names:
            if item == 'span':
                namelist.append('Span (m)')
            elif item == 'rcp_tip':
                namelist.append('Tip taper (%)')
            elif item == 'sweep':
                namelist.append('1/4 chord sweep (deg)')
            elif item == 'dihedral':
                namelist.append('Dihedral (deg)')
            elif item == 'twist_tip':
                namelist.append('Twist (deg)')

        for i in range(0,np.shape(x1)[0]): #x1
            for j in range(0,np.shape(x2)[1]): #x2
                point = [x1[i,j],x2[i,j]]
                gen[i,j] = model.predict(point)
#        print gen
    
        if zlabel.lower() in ['lift','ld','l/d']:
            gen = -gen


        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(x1, x2, gen, rstride=1, cstride=1,cmap=mapname, edgecolor='none',alpha=0.7)
        ax.set_xlabel(namelist[0])
        ax.set_ylabel(namelist[1])
        ax.set_zlabel(zlabel)
        ax.set_aspect('auto')
        plt.show()

        if not model1 == None:
            gen1 = np.zeros(np.shape(x1))
            for i in range(0,np.shape(x1)[0]):
                for j in range(0,np.shape(x2)[1]): #x2
                    point = [x1[i,j],x2[i,j]]
                    gen1[i,j] = model1.predict(point)

            if zlabel.lower() in ['lift','ld','l/d']:
                gen1 = -gen1

            diff = np.divide((np.array(gen) - np.array(gen1)),gen)

            fig1 = plt.figure()
            ax1 = plt.axes(projection='3d')
            ax1.plot_surface(x1, x2, gen1, rstride=1, cstride=1,cmap=mapname, edgecolor='none',alpha=0.7)
            ax1.set_xlabel(namelist[0])
            ax1.set_ylabel(namelist[1])
            ax1.set_zlabel(zlabel)
            ax1.set_aspect('auto')
            plt.show()
            print diff
            fig2 = plt.figure()
            ax2 = plt.axes(projection='3d')
            ax2.plot_surface(x1, x2, diff, cmap='jet',rstride=1, cstride=1, edgecolor='none',alpha=0.7)
            ax2.set_xlabel(namelist[0])
            ax2.set_ylabel(namelist[1])
            ax2.set_zlabel(zlabel)
            ax2.set_aspect('auto')
            plt.show()

            fig3 = plt.figure()
            ax3 = plt.axes(projection='3d')
            ax3.plot_surface(x1, x2, gen, cmap='winter',rstride=1, cstride=1, edgecolor='none',alpha=0.5)
            ax3.plot_surface(x1, x2, gen1, cmap='copper',rstride=1, cstride=1, edgecolor='none',alpha=0.5)
            ax3.set_xlabel(namelist[0])
            ax3.set_ylabel(namelist[1])
            ax3.set_zlabel(zlabel)
            ax3.set_aspect('auto')
            plt.show()



        return 'done'





    def get_plot3X(self,nexus,model=None,model1=None,zlabel='',mapname='viridis'):
        """
            Somehow I can do many other things, but I cannot plot things. 
            I am very stupid
        """
        from mpl_toolkits import mplot3d
#        matplotlib inline
        import matplotlib.pyplot as plt

        inputs      = nexus.optimization_problem.inputs

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

        # this should always perform, but in case is already ndarray:
        if isinstance(ub, np.ndarray) == False:
            ub      = np.array(ub)
            lb      = np.array(lb)

        size = 50

        lx1  = np.linspace(lb[0],ub[0],size)
        lx2  = np.linspace(lb[1],ub[1],size)
        lx3  = np.linspace(lb[2],ub[2],size)
        
        x1, x2, x3 = np.meshgrid(lx1,lx2,lx3)
        
        gen = np.zeros(np.shape(x1))

        for i in range(0,np.shape(x1)[0]): #x1
            for j in range(0,np.shape(x2)[1]): #x2
                point = [x1[i,j],x2[i,j],x3[i,j]]
                gen[i,j] = model.predict(point)
        print gen

        namelist = []
        for item in names:
            if item == 'span':
                namelist.append('Span (m)')
            elif item == 'rcp_tip':
                namelist.append('Tip taper (%)')
            elif item == 'sweep':
                namelist.append('1/4 chord sweep (deg)')
            elif item == 'dihedral':
                namelist.append('Dihedral (deg)')
            elif item == 'twist_tip':
                namelist.append('Twist (deg)')

        for i in range(0,np.shape(x1)[0]): #x1
            for j in range(0,np.shape(x2)[1]): #x2
                point = [x1[i,j],x2[i,j]]
                gen[i,j] = model.predict(point)
#        print gen
    
        if zlabel.lower() in ['lift','ld','l/d']:
            gen = -gen


        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(x1, x2, gen, rstride=1, cstride=1,facecolors=fcolors, edgecolor='none',alpha=0.7)
        ax.set_xlabel(namelist[0])
        ax.set_ylabel(namelist[1])
        ax.set_zlabel(zlabel)
        cax, _ = matplotlib.colorbar.make_axes(ax)
        cmap = matplotlib.cm.get_cmap(mapname)
        cbar = matplotlib.colorbar.ColorbarBase(cax,cmap=cmap,norm=norm)
        cbar.set_label(namelist[2],rotation=270)
        plt.show()

        if not model1 == None:
            gen1 = np.zeros(np.shape(x1))
            for i in range(0,np.shape(x1)[0]):
                for j in range(0,np.shape(x2)[1]): #x2
                    point = [x1[i,j],x2[i,j]]
                    gen1[i,j] = model1.predict(point)

            if zlabel.lower() in ['lift','ld','l/d']:
                gen1 = -gen1

            diff = np.divide((np.array(gen) - np.array(gen1)),gen)

            fig1 = plt.figure()
            ax1 = plt.axes(projection='3d')
            ax1.plot_surface(x1, x2, gen1, rstride=1, cstride=1,cmap=mapname, edgecolor='none',alpha=0.7)
            ax1.set_xlabel(namelist[0])
            ax1.set_ylabel(namelist[1])
            ax1.set_zlabel(zlabel)
#            plt.show()
            print diff
            fig2 = plt.figure()
            ax2 = plt.axes(projection='3d')
            ax2.plot_surface(x1, x2, diff, cmap='jet',rstride=1, cstride=1, edgecolor='none',alpha=0.7)
            ax2.set_xlabel(namelist[0])
            ax2.set_ylabel(namelist[1])
            ax2.set_zlabel(zlabel)
            plt.show()

        return 'done'







    #   Kriging model point checker
    def kriging_model_check(self,k):
        lr,lc           = np.shape(k.X)[0],np.shape(k.X)[1]
        X_c, y_c        = np.zeros((lr-1, lc)), np.zeros(lr-1)
        # time is of the essence
        if lr < 8:
            sz        = 3
        elif lr > 12:
            sz        = 1
        else:
            sz        = 2
        test_points     = np.random.randint(lr,size=sz)  # just three pts
        
        #   check several randomly chosen point sets
        for i in range(0,len(test_points)):
            a            = test_points[i]
            test_X       = k.X[a,:]
            compar_y     = k.y[a]
            print 'kriging check : a : ' + str(a)
            # make comparison models
            if a != 0 or a != lr:
                X_c[0:a/2,:]    = k.X[0:a/2,:]
                X_c[a/2:a-1,:]  = k.X[a/2+1:a,:]
                y_c[0:a/2]      = k.y[0:a/2]
                y_c[a/2:a-1]    = k.y[a/2+1:a]
            
            elif a == 0:
                X_c[:,:]        = k.X[1:a,:]
                y_c[:]          = k.y[1:a]
            else:
                X_c[:,:]        = k.X[0:a-1,:]
                y_c[:]          = k.y[0:a-1]
            # compare
            k_c             = kriging(X_c, y_c)
            k_c.train()
            test_y          = k_c.predict(test_X)
            # these ifs prevent infs :/
            if compar_y != 0 and test_y != 0:
                frac_check      = np.abs((compar_y-test_y)/compar_y)
            elif compar_y == 0 and test_y != 0:
                frac_check      = np.abs((compar_y-test_y)/test_y)
            else:
                frac_check = 0
            # update holder numbers
            if i == 0:
                frac        = frac_check
            if frac_check > frac:
                frac        = frac_check
            print 'kriging check : largest percentage difference : ' + str(frac)
        return frac








    def save_surrogates(self):
        mods = 0


        if not self.model0 == None:
            mods = mods+1
        if not self.model1 == None:
            mods = mods+1
        if not self.model2 == None:
            mods = mods+1
        for i in range(0,mods+1):
            X   = self.X
            y   = self.y
            with open('./' + 'modeldata'+str(i)+'.csv','w+b') as savefile:
                wr = csv.writer(savefile, quoting = csv.QUOTE_ALL)
                for j in range(len(y)):
                    row = X[i]
                    row.append(y[i])
                    wr.writerow(row)
                print 'model data for model'+str(i) + ' write complete'




if __name__ == '__main__':
    s = Surrogate_Data()
#    s.get_ck_plot('adf')
    s.ck_from_file(nexus,'./results/LFMF_earth_sptp_ckcompar.csv')
#    print "dook"
#    sp  = samplingplan(2)
#    Xc  = sp.rlh(10)
#    Xe  = np.array(random.sample(Xc,6)    )

#    print Xc
#    print Xe

