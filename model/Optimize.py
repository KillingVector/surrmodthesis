# Optimize.py
# 
# Created:  Feb 2016, E. Botero
# Modified: 

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

from SUAVE.Core import Units, Data
import numpy as np
import Vehicles_lf, Vehicles_lf, Vehicles_hf
import Analyses_lf, Analyses_lf, Analyses_hf
import Missions
import Procedure
#import Plot_Mission
import SUAVE.Optimization.Package_Setups.scipy_setup as scipy_setup
import SUAVE.Optimization.Package_Setups.pyopt_setup as pyopt_setup
from Surrogate_Nexus import Nexus

from pyKriging import saveModel, loadModel
from pyKriging.krige import kriging  

#from SUAVE.Attributes.Solids import Aluminium, Bidirectional_Carbon_Fiber, Unidirectional_Carbon_Fiber

from surrogate import Surrogate_Data
from inspyred_wrapper import inspyred_wrapper as optimizer
import openmdao
from openmdao.surrogate_models.kriging import KrigingSurrogate


import csv, datetime, os

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# ----------------------------------------------------------------------        
#   Run the whole thing
# ----------------------------------------------------------------------  

# place holder location for design variables

def main():
    t1 = datetime.datetime.now()
    
    '''    Modify these values to choose fidelity level
        and modeling method    '''
    # FIDELITY LEVEL: low = 0;  med = 1;  high = 2
    fidelity_level  = 2 # 0 or 2 atm.
    # MODEL METHOD
    model_method    = 'k'  # k or ck
    # ALLOWED TIME (s) 
    hours           = 1.
    mins            = 30.
    secs            = 0.
    time            = hours*60*60 + mins*60 + secs
    # 
    fidelity_method = [fidelity_level, model_method]
    from_file = True
    
    nexus = setup(fidelity_method)
    optprob = nexus.optimization_problem
    surr = nexus.surrogate_data
#    surr.plot_from_file(nexus,filename)
#    quit()

    inpstr = optprob.inputs[:,0]
    objstr = optprob.objective[:,0]
    print "inputs" + str(inpstr)
    print 'object' + str(objstr)

    inputstring=''
    for var in inpstr:
        if not inputstring == '':
            if var == 'rcp_tip':
                contr = 'tp'
            elif var == 'dihedral':
                contr = 'do'
            elif var == 'span':
                contr = 'sp'
            else:
                contr = var[0:2]
            inputstring = inputstring+contr
        else:
            inputstring = inputstring+var
    savename = model_method+str(fidelity_level)+'-' + inputstring+'-'
    print 'Files will be saved under prefix : \"' + savename +'\"'

    t1b = datetime.datetime.now()

    print optprob.inputs[:,1]
#    optprob.inputs[:,1] = [4.3301653087,	0.0504995492,	26.7446129654,	-1.9490900158]

#    print nexus.objective()
#    quit()
    
##########################################################################



    if model_method == 'k' and not from_file:
#        a = nexus.objective()
        surr.sample_plan.size = 40
        surr.sample_plan.lhc_type   = 'o'
        surr.sample_plan.time       = time
#        surr.create_sample(nexus)   # DO THIS JUST FOR CORNERS
        timea = datetime.datetime.now()

        data    = np.genfromtxt('/home/ashaiden/Documents/surrmodthesis/model/rawresults/LHC/30degk0-spantpswtw-LHC.csv',delimiter=',')
#        surr.sample_plan.lhc = data[0:6,:]
### 1
#        surr.evaluate_of(nexus)
#        t1b = datetime.datetime.now()
#        with open('./rawresults/kriging/30degaoa'+savename+str(t1b)+'.csv','w+b') as file11:
##        with open('/home/ashaiden/Documents/surrmodthesis/model/rawresults/LHC/30degk0-spantpswtw-LHC.csv', 'w+b') as file11:
#            wr11=csv.writer(file11)
#            for i in range(0,np.shape(nexus.summary.aoa)[0]):
##            for i in range(0,np.shape(surr.sample_plan.lhc)[0]):
#                row = []
##                for item in surr.sample_plan.lhc[i,:]:
#                for item in nexus.summary.aoa[i]:
#                    row.append(item)
#                wr11.writerow(row)

##        quit()
#        with open('./rawresults/kriging/30deg'+savename+'X-y'+str(t1b)+'.csv','w+b') as file12:
#            wr12=csv.writer(file12)
#            for i in range(0,np.shape(surr.X)[0]):
#                row = []
#                for item in surr.X[i,:]:
#                    row.append(item)
#                row.append(surr.y[i,0])
#                row.append(surr.y[i,1])
#                wr12.writerow(row)
#        nexus.summary.aoa = []

#### 2
#        surr.sample_plan.lhc = data[6:12,:]

#        surr.evaluate_of(nexus)
#        t1b = datetime.datetime.now()
#        with open('./rawresults/kriging/30degaoa'+savename+str(t1b)+'.csv','w+b') as file21:
#            wr21=csv.writer(file21)
#            for i in range(0,np.shape(nexus.summary.aoa)[0]):
#                row = []
#                for item in nexus.summary.aoa[i]:
#                    row.append(item)
#                wr21.writerow(row)

#        with open('./rawresults/kriging/30deg'+savename+'X-y'+str(t1b)+'.csv','w+b') as file22:
#            wr22=csv.writer(file22)
#            for i in range(0,np.shape(surr.X)[0]):
#                row = []
#                for item in surr.X[i,:]:
#                    row.append(item)
#                row.append(surr.y[i,0])
#                row.append(surr.y[i,1])
#                wr22.writerow(row)
#        nexus.summary.aoa = []

#### 3
#        surr.sample_plan.lhc = data[12:18,:]

#        surr.evaluate_of(nexus)
#        t1b = datetime.datetime.now()
#        with open('./rawresults/kriging/30degaoa'+savename+str(t1b)+'.csv','w+b') as file31:
#            wr31=csv.writer(file31)
#            for i in range(0,np.shape(nexus.summary.aoa)[0]):
#                row = []
#                for item in nexus.summary.aoa[i]:
#                    row.append(item)
#                wr31.writerow(row)

#        with open('./rawresults/kriging/30deg'+savename+'X-y'+str(t1b)+'.csv','w+b') as file32:
#            wr32=csv.writer(file32)
#            for i in range(0,np.shape(surr.X)[0]):
#                row = []
#                for item in surr.X[i,:]:
#                    row.append(item)
#                row.append(surr.y[i,0])
#                row.append(surr.y[i,1])
#                wr32.writerow(row)
#        nexus.summary.aoa = []

#### 4

#        surr.sample_plan.lhc = data[18:24,:]

#        surr.evaluate_of(nexus)
#        t1b = datetime.datetime.now()
#        with open('./rawresults/kriging/30degaoa'+savename+str(t1b)+'.csv','w+b') as file41:
#            wr41=csv.writer(file41)
#            for i in range(0,np.shape(nexus.summary.aoa)[0]):
#                row = []
#                for item in nexus.summary.aoa[i]:
#                    row.append(item)
#                wr41.writerow(row)

#        with open('./rawresults/kriging/30deg'+savename+'X-y'+str(t1b)+'.csv','w+b') as file42:
#            wr42=csv.writer(file42)
#            for i in range(0,np.shape(surr.X)[0]):
#                row = []
#                for item in surr.X[i,:]:
#                    row.append(item)
#                row.append(surr.y[i,0])
#                row.append(surr.y[i,1])
#                wr42.writerow(row)
#        nexus.summary.aoa = []

### 5

#        surr.sample_plan.lhc = data[23:30,:]

#        surr.evaluate_of(nexus)
#        t1b = datetime.datetime.now()
#        with open('./rawresults/kriging/30degaoa'+savename+str(t1b)+'.csv','w+b') as file51:
#            wr51=csv.writer(file51)
#            for i in range(0,np.shape(nexus.summary.aoa)[0]):
#                row = []
#                for item in nexus.summary.aoa[i]:
#                    row.append(item)
#                wr51.writerow(row)

#        with open('./rawresults/kriging/30deg'+savename+'X-y'+str(t1b)+'.csv','w+b') as file52:
#            wr52=csv.writer(file52)
#            for i in range(0,np.shape(surr.X)[0]):
#                row = []
#                for item in surr.X[i,:]:
#                    row.append(item)
#                row.append(surr.y[i,0])
#                row.append(surr.y[i,1])
#                wr52.writerow(row)
#        nexus.summary.aoa = []

#### 6

#        surr.sample_plan.lhc = data[30:38,:]

#        surr.evaluate_of(nexus)
#        t1b = datetime.datetime.now()
#        with open('./rawresults/kriging/30degaoa'+savename+str(t1b)+'.csv','w+b') as file61:
#            wr61=csv.writer(file61)
#            for i in range(0,np.shape(nexus.summary.aoa)[0]):
#                row = []
#                for item in nexus.summary.aoa[i]:
#                    row.append(item)
#                wr61.writerow(row)

#        with open('./rawresults/kriging/30deg'+savename+'X-y'+str(t1b)+'.csv','w+b') as file62:
#            wr62=csv.writer(file62)
#            for i in range(0,np.shape(surr.X)[0]):
#                row = []
#                for item in surr.X[i,:]:
#                    row.append(item)
#                row.append(surr.y[i,0])
#                row.append(surr.y[i,1])
#                wr62.writerow(row)
#        nexus.summary.aoa = []

#### 7
#        surr.sample_plan.lhc = data[38:44,:]

#        surr.evaluate_of(nexus)
#        t1b = datetime.datetime.now()
#        with open('./rawresults/kriging/30degaoa'+savename+str(t1b)+'.csv','w+b') as file71:
#            wr71=csv.writer(file71)
#            for i in range(0,np.shape(nexus.summary.aoa)[0]):
#                row = []
#                for item in nexus.summary.aoa[i]:
#                    row.append(item)
#                wr71.writerow(row)

#        with open('./rawresults/kriging/30deg'+savename+'X-y'+str(t1b)+'.csv','w+b') as file72:
#            wr72=csv.writer(file72)
#            for i in range(0,np.shape(surr.X)[0]):
#                row = []
#                for item in surr.X[i,:]:
#                    row.append(item)
#                row.append(surr.y[i,0])
#                row.append(surr.y[i,1])
#                wr72.writerow(row)
#        nexus.summary.aoa = []

### 8

        surr.sample_plan.lhc = data[44:57,:]

        surr.evaluate_of(nexus)
        t1b = datetime.datetime.now()
        with open('./rawresults/kriging/30degaoa'+savename+str(t1b)+'.csv','w+b') as file81:
            wr81=csv.writer(file81)
            for i in range(0,np.shape(nexus.summary.aoa)[0]):
                row = []
                for item in nexus.summary.aoa[i]:
                    row.append(item)
                wr81.writerow(row)

        with open('./rawresults/kriging/30deg'+savename+'X-y'+str(t1b)+'.csv','w+b') as file82:
            wr82=csv.writer(file82)
            for i in range(0,np.shape(surr.X)[0]):
                row = []
                for item in surr.X[i,:]:
                    row.append(item)
                row.append(surr.y[i,0])
                row.append(surr.y[i,1])
                wr82.writerow(row)
        nexus.summary.aoa = []

#        surr.X = data[:,0:3]
#        surr.y = data[:,3:5]
        timeb = datetime.datetime.now()
        print 'the run time was : ' + str((timeb - timea).total_seconds())

        quit()



#        surr.single_fid_kriging(nexus, improve=False)

        print '\n\n Kriging model info:\n'
        print 'Thetas L/D : ' + str(surr.model0.theta)
        print 'Thetas mass: ' + str(surr.model1.theta)
        print 'p L/D : ' + str(surr.model0.pl)
        print 'p mass: ' + str(surr.model1.pl)

#        k0 = surr.model0
#        if k0.pl[0] < 1.5:
#            print 'cor1'
#            k0.pl[0] = 2.
#        elif k0.pl[1] < 1.5:
#            print 'cor2'
#            k0.pl[1] = 2.

#        k1 = surr.model1
#        if len(inpstr) <= 2:
#            surr.get_plot(nexus,model = k0,zlabel='L/D',mapname='winter')
#            surr.get_plot(nexus,model = k1,zlabel='Mass(kg)',mapname='copper')
#        elif len(inpstr) == 3:
#            surr.get_plot3X(nexus,model=k1,zlabel='L/D',mapname='winter')
##            surr.get_plot3X(nexus,model=k2,zlabel='Mass (kg)',mapname='copper')
#        saveModel(surr.model0,'./rawresults/kriging/'+savename+str(np.shape(surr.X))+'-LD'+'.pkl')
#        saveModel(surr.model1,'./rawresults/kriging/'+savename+str(np.shape(surr.X))+'-mass'+'.pkl')
        t2b = datetime.datetime.now()
#        quit()
#        with open('./rawresults/kriging/'+savename+'X-y'+'.csv','w+b') as file4:
#            wr4=csv.writer(file4)
#            for i in range(0,np.shape(surr.X)[0]):
#                row = []
#                for item in surr.X[i,:]:
#                    row.append(item)
#                row.append(surr.y[i,0])
#                row.append(surr.y[i,1])
#                wr4.writerow(row)



#        quit()

        t2 = datetime.datetime.now()
        iw = optimizer(nexus,'k')
#        quit()
        iw.evolve()
        t3 = datetime.datetime.now()
        iw.show_results(title='Pareto-front')
        iw.show_gen_results()
        iw.save_results(name = './rawresults/kriging/'+savename+str(np.shape(surr.X))+'_opt')


        print 'For LHC size: ' + str(np.shape(surr.X))
        print 'Setup time = ' + str((t1b-t1).total_seconds()) + ' sec'
        print 'Model runs time = ' + str((t2a-t1b).total_seconds()) + ' sec'
        print 'Optimisation time = ' + str((t3-t2b).total_seconds()) + ' sec'

        print '\n\n Kriging model info:\n'
        print 'Thetas L/D : ' + str(surr.model0.theta)
        print 'Thetas mass: ' + str(surr.model1.theta)
        print 'p L/D : ' + str(surr.model0.pl)
        print 'p mass: ' + str(surr.model1.pl)

#####################################################################
    elif model_method == 'ck' and not from_file:


        surr.sample_plan.lhc_type   = 'o'
        surr.sample_plan.time   = time
        t2a = surr.hybrid_cokriging(nexus,'./rawresults/cokriging'+savename)
#        surr.save_ck('./results/'+savename+'cokriging') # should be unnecessary
        t2 = datetime.datetime.now()
        # plotting methods
        if len(inpstr) <= 2:
            try:
                surr.get_plot(nexus,model=surr.model1, model1=surr.modelck0,zlabel='L/D') # mod1 is l/d
            except:
                print 'no second l/d model?'
                try:
                    surr.get_plot(nexus,model=surr.modelck0) # just plot ck model
                except:
                    print 'no models?'
        iw = optimizer(nexus,'ck')
        iw.evolve()
        iw.show_results(title='')
        iw.show_gen_results()
        iw.save_results(name = './rawresults/cokriging/'+savename+'opt')
#    quit()


######################### FROM FILE
    elif model_method == 'ck' and from_file:

        # surr.load_ck() # no .pkl
        # surr.model999 = loadModel() # pykriging built, need .pkl
        surr.load_ck('/home/ashaiden/Documents/surrmodthesis/model/rawresults/cokrigingck2-spantp-corr-rho')
#        surr.load_ck('/home/ashaiden/Documents/surrmodthesis/model/rawresults/cokrigingck2-spantpsw-')
#        surr.load_ck('/home/ashaiden/Documents/surrmodthesis/model/rawresults/cokrigingck2-spantpsw-(17, 3)')
        data1 = np.genfromtxt('/home/ashaiden/Documents/surrmodthesis/model/rawresults/kriging/kriging_XY/k0-sptp-X-y.csv',delimiter=',')
        data2 = np.genfromtxt('/home/ashaiden/Documents/surrmodthesis/model/rawresults/kriging/kriging_XY/k2-sptp-X-ySML.csv',delimiter=',')
#        surr.model0 = kriging(data1[:,0:3],data1[:,4])
#        surr.model0.train()
        ck0 = surr.modelck0
##        print ck0.predict([4.28,0.05,45])
##        print ck0.predict([13.45,0.065,5.25])
#        print surr.model0.predict([20., 0.05, 45.])

        

        k0 = kriging(data1[:,0:2],data1[:,2])
        k0.train()

        k1 = kriging(data2[:,0:2],data2[:,2])
        k1.train()
#        surr.model1 = k1
        t2 = datetime.datetime.now()
#        a = ck0.predict([7.8,0.38])
#        b = k1.predict([7.8,0.38])

#        surr.get_plot(nexus,model=ck0,zlabel='L/D',mapname='winter')

        if len(inpstr) <= 2:
            surr.get_plot(nexus,model=k0,model1=ck0,zlabel='L/D',mapname='winter')#model1=k1,
            surr.get_plot(nexus,model=k1,model1=ck0,zlabel='L/D',mapname='winter')
##            surr.get_plot(nexus,model=k1,model1=ck0,zlabel='L/D')
##            surr.get_plot(nexus,model=ke,model1=surr.modelck0,zlabel='L/D')

        quit()
        iw = optimizer(nexus,'ck')
        iw.evolve()
        iw.show_results(title='')
        iw.show_gen_results()
#        iw.save_results(name = './rawresults/cokriging/ck2-'+inputstring+str(np.shape(ck0.Xe)))
        iw.save_results(name = './rawresults/kriging/aaSPTP-ck0-'+inputstring)

######################### FROM FILE
    elif model_method == 'k' and from_file:

        p1  = [6., .4, 30.]
        p2  = [12., .6, 15.]
        p3  = [17., .2, 47.]
        p4  = [8., .8, 8.]

        
#        data1 = np.genfromtxt('/home/ashaiden/Documents/surrmodthesis/model/rawresults/kriging/1NEW30degmax_sptpswptw_results.csv',delimiter=',')
##        data2 = np.genfromtxt('/home/ashaiden/Documents/surrmodthesis/model/rawresults/kriging/kriging_XY/k2-sptp-X-ySML.csv',delimiter=',')
#        data2 = data1
#        k0 = kriging(data1[:,0:4],data1[:,4])
#        k0.train()

#        k1 = kriging(data2[:,0:4],data2[:,5])
#        k1.train()

#        saveModel(k0,'/home/ashaiden/Documents/surrmodthesis/model/rawresults/kriging/0NEW30degmax_sptpswpTW_results-k2.pkl')
#        saveModel(k1,'/home/ashaiden/Documents/surrmodthesis/model/rawresults/kriging/0NEW30degmax_sptpswpTW_results-k2mass.pkl')

        k0 = loadModel('/home/ashaiden/Documents/surrmodthesis/model/rawresults/kriging/k2-spantpsw-(37, 3)-LD.pkl')
        k1 = loadModel('/home/ashaiden/Documents/surrmodthesis/model/rawresults/kriging/k2-spantpsw-(37, 3)-mass.pkl')
        
#        surr.get_plot(nexus,model=k0,zlabel='L/D',mapname='winter')
#        surr.get_plot(nexus,model=k1,zlabel='Mass (kg)',mapname='copper')        

#        quit()
#        surr.get_plot(nexus,model=k0,model1=ck0,zlabel='L/D',mapname='winter')#model1=k1,
#        surr.get_plot(nexus,model=k1,model1=ck0,zlabel='L/D',mapname='winter')

        surr.model0 = k0
        surr.model1 = k1


        iw = optimizer(nexus,'k')
        iw.evolve()
        iw.show_results(title='')
        iw.show_gen_results()
#        iw.save_results(name = './rawresults/kriging/0NEW30degmax_k2-'+inputstring)

        quit()
        print '\n\n Kriging model info:\n'
        print 'For LHC size: ' + str('12,3')
        print 'Thetas L/D : ' + str(m10.theta)
        print 'Thetas mass: ' + str(k1m.theta)
        print 'p L/D : ' + str(m10.pl)
        print 'p mass: ' + str(k1m.pl)

        print 'For LHC size: ' + str('18,3')
        print 'Thetas L/D : ' + str(m20.theta)
        print 'Thetas mass: ' + str(k2m.theta)
        print 'p L/D : ' + str(m20.pl)
        print 'p mass: ' + str(k2m.pl)


#        optprob.inputs[:,1] = p1
#        p1o = nexus.objective()

#        optprob.inputs[:,1] = p2
#        p2o = nexus.objective()
#        optprob.inputs[:,1] = p3
#        p3o = nexus.objective()

#        optprob.inputs[:,1] = p4
#        p4o = nexus.objective()
#        print p4o
#        quit()

        print '==='+str(p1)
        print 'corel: ' + str(-8.758738709620385)
        print m10.predict(p1)
        print m20.predict(p1)
#        print m30.predict(p1)
#        print m40.predict(p1)
#        print m50.predict(p1)
#        
        print '==='+str(p2)
        print 'corel: ' + str(-1.7228475920765616)
        print m10.predict(p2)
        print m20.predict(p2)
#        print m30.predict(p2)
#        print m40.predict(p2)
#        print m50.predict(p2)

        print '==='+str(p3)
        print 'corel: ' + str(-4.1441922666)
        print m10.predict(p3)
        print m20.predict(p3)
#        print m30.predict(p3)
#        print m40.predict(p3)
#        print m50.predict(p3)

        print '==='+str(p4)
        print 'corel: ' + str(-5.2278043817745)
        print m10.predict(p4)
        print m20.predict(p4)
#        print m30.predict(p4)
#        print m40.predict(p4)
#        print m50.predict(p4)





   
    return

# ----------------------------------------------------------------------        
#   Inputs, Objective, & Constraints
# ----------------------------------------------------------------------  

def base_design():

    # vehicle dats (:
    vec             = Data()

    vec.span            = 9.36128008739618#6.5
    vec.root_chord      = 2.14

    # size payload (max)
    # payload used to stabilise
    payload         = Data()
    payload.payload_mass    = 50.
    payload.static_margin   = .1
    # volume dimensions
    pl_len      = 1.65  # m
    pl_wid      = .55   # m
    pl_hgt      = .55   # m
    # make pc chord and span vals
    pl_psl      = pl_wid/vec.span
    pl_ttc      = pl_hgt/vec.root_chord
#    print pl_psl

    # make sizing vectors
    vec.psl             = np.array([0.])#, pl_psl])#, 0.5])
    vec.sqc             = np.array([27.4014833295441])#30.])#, 5.])#, 10.])
    vec.rcp             = np.array([1., 0.0504995491847691])#, .4])#, .1]) #rcp[-1] tip chord
    vec.ttc             = np.array([pl_ttc, 0.2])#, 0.1])#, .08]) # thickness
    vec.do              = np.array([0.])#, 0.])#, 0.])
    vec.tw              = np.array([0.,-0.925145936653042]) # tip twist only, root always 0 for flying wing
    vec.num_seg         = len(vec.psl)
    

    # Choose material

    material        = Data()
    material.Exx    = 70.0e9
    material.Eyy    = 70.0e9
    material.G      = 26e9
    material.rho    = 2700.
    material.poisson= 0.35
    material.sig_c  = 200.0e6
    material.sig_t  = 300.0e6
    opt_names       = ['span','root_chord']

    vec.aerofoil        = None
    #full vec of names:
    #['span','root_chord','psl','sle','rcp','do','tw']
    
    return vec, payload, material, opt_names

def setup(fidelity_method):

    [fidelity_level, model_method] = fidelity_method

    nexus                        = Nexus()
    problem                      = Data()
    nexus.optimization_problem   = problem

    surrogate_data               = Surrogate_Data()
    vec, payload, material, opt_names= base_design()

    surrogate_data.vec.init_vec  = vec
    surrogate_data.vec.payload   = payload
    surrogate_data.vec.material  = material
    surrogate_data.vec.opt_names = opt_names

    nexus.surrogate_data        = surrogate_data

    # -------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------
#    problem.fidelity    = fidelity_level
    nexus.surrogate_data.fidelity_level = fidelity_level
    # [ tag , initial, [lb,ub], scaling, units ]
    
    # check wing segmentation
    if len(vec.psl) == 1:
        problem.inputs      = np.array([
            [ 'span'       ,   vec.span, (  2*vec.root_chord,   20.0 ),  1.0, Units.less],
#    #        [ 'rootChord',   vec.root_chord, (  0.5,    10. ),  1.0, Units.meter  ],
            [ 'rcp_tip' ,  vec.rcp[-1], (  0.05,    1.0 ),  1.0, Units.less    ],  
            [ 'sweep', vec.sqc[0], (  0.0,   30.0 ),  1.0, Units.degrees ],
#            [ 'dihedral' ,  vec.do[0] , ( -5.0,    5.0 ),  1.0, Units.degrees ],
#            [ 'twist_tip' ,  vec.tw[-1], ( -5.0,    5.0 ),1.0, Units.degrees  ],
        ])

    elif len(vec.psl) > 1:
        problem.inputs      = np.array([
    #        [ 'span'       ,   vec.span, (  1.0,   20.0 ),  1.0, Units.less],
    ##        [ 'rootChord',   vec.root_chord, (  0.5,    10. ),  1.0, Units.meter  ],
    ##        [ 'psl_s1'  ,   vec.psl[1], (  0.2,    0.99),  1.0, Units.less    ],
    #        [ 'psl_s2'  ,   vec.psl[2], (  0.2,    0.99),  1.0, Units.less    ],
    ##        [ 'rcp_s1'  ,   vec.rcp[1], (  0.1,    1.0 ),  1.0, Units.less    ],
    #        [ 'rcp_s2'  ,   vec.rcp[2], (  0.1,    1.0 ),  1.0, Units.less    ],
    #        [ 'rcp_tip' ,  vec.rcp[-1], (  0.1,    1.0 ),  1.0, Units.less    ],  
            [ 'swp_s0', vec.sqc[0], (  0.0,   60.0 ),  1.0, Units.degrees ],
    #        [ 'swp_s1', vec.sqc[1], (  0.0,   60.0 ),  1.0, Units.degrees ],
    #        [ 'swp_s2', vec.sqc[2], (-10.0,   50.0 ),  1.0, Units.degrees ],
    #        [ 'do_s1' ,  vec.do[0] , ( -5.0,    5.0 ),  1.0, Units.degrees ],
    #        [ 'do_s2' ,  vec.do[1] , ( -5.0,    5.0 ),  1.0, Units.degrees ],
            [ 'twist_tip' ,  vec.tw[-1], ( -5.0,    5.0 ),1.0, Units.degrees  ],
        ])

    # -------------------------------------------------------------------
    # Objective
    # -------------------------------------------------------------------

    # [ tag, scaling, units ]
    problem.objective   = np.array([
         [ 'obj_1', 1. , Units.less], # inverse D/L + penalty
         [ 'obj_mass'  , 1. , Units.kg  ],
    ])
#    surrogate.objective = problem.objective
    # -------------------------------------------------------------------
    # Constraints
    # -------------------------------------------------------------------

    # [ tag, sense, edge, scaling, units ]
    problem.constraints = np.array([
#         [ 'wing_span'   , '>', 10.0, 1.0, Units.meter],
         [ 'total_mass'  , '>', 1500., 1.0, Units.kg],
    ])
#    surrogate.constraints = problem.constraints
    # -------------------------------------------------------------------
    #  Aliases
    # -------------------------------------------------------------------
    # [ 'alias' , ['data.path1.name','data.path2.name'] ]
    problem.aliases = [
        [ 'rootChord','vehicle_configurations.base.wings.main_wing.chords.root' ],
        [ 'span'                , ['vehicle_configurations.base.wings.main_wing.spans.projected']           ], 
# =============== Single section wing
        [ 'sweep', 'vehicle_configurations.base.wings.main_wing.sweeps.quarter_chord'],
        [ 'didhedral','vehicle_configurations.base.wings.main_wing.dihedral'],
        [ 'taper', 'vehicle_configurations.base.wings.main_wing.taper'],
# =============== Shared: twists
        [ 'twist_root'          , 'vehicle_configurations.base.wings.main_wing.twists.root'               ],
        [ 'twist_tip'           , 'vehicle_configurations.base.wings.main_wing.twists.tip'                ],
# =============== Multiple sections
#       PERCENT SPAN LOCATION
        [ 'psl_s1', 'vehicle_configurations.base.wings.main_wing.Segments.section_1.percent_span_location'],
        [ 'psl_s2', 'vehicle_configurations.base.wings.main_wing.Segments.section_2.percent_span_location'],
#       SEGMENT SWEEP
        [ 'swp_s0', 'vehicle_configurations.base.wings.main_wing.Segments.section_0.sweeps.quarter_chord'],
        [ 'swp_s1', 'vehicle_configurations.base.wings.main_wing.Segments.section_1.sweeps.quarter_chord'],
        [ 'swp_s2', 'vehicle_configurations.base.wings.main_wing.Segments.section_2.sweeps.quarter_chord'],
#       SEGMENT DIHEDRAL
        [ 'do_s1' , 'vehicle_configurations.base.wings.main_wing.Segments.section_1.dihedral_outboard'],
        [ 'do_s2' , 'vehicle_configurations.base.wings.main_wing.Segments.section_2.dihedral_outboard'],
#       SEGMENT TAPERS
        [ 'rcp_tip' , 'vehicle_configurations.base.wings.main_wing.taper' ],
        [ 'rcp_s2'  , 'vehicle_configurations.base.wings.main_wing.Segments.section_2.root_chord_percent'],

# ====== TARGET VALUES
        [ 'total_mass'    , 'summary.mass'            ],
        [ 'obj_1'        , 'summary.obj_ld'          ],
        [ 'obj_mass'      , 'summary.obj_mass'        ],
    ]       


    # Decide method process    
    cokrig_strings  = ['ck','cokriging','cokrig','co-kriging','cookie']
    pick_again      = '\n Expected fidelity value in fidelity_method between 0 and 1\n Please select a value in: 0 - low, 1 - med, 2 - high'
    
    if fidelity_level > 0 and model_method.lower() in cokrig_strings:
        print 'method route 1'
        for i in range(0,fidelity_method[0]+1):
            if i == 0:
                print 'low fid'
                nexus_0           = Nexus()
                nexus_0.fidelity_level         = 0
                nexus.surrogate_data.op.lf     = nexus_0
                nexus_0.surrogate_data         = surrogate_data # shot myself in the foot here :/

                nexus_0.optimization_problem   = problem
                nexus_0.vehicle_configurations = Vehicles_lf.setup(nexus_0)
                nexus_0.analyses  = Analyses_lf.setup(nexus_0.vehicle_configurations)
                nexus_0.missions  = Missions.setup(nexus_0.analyses,nexus_0.vehicle_configurations)
                nexus_0.procedure = Procedure.setup(nexus_0)
            if i == 1:
                print 'med fid'
                nexus_1           = Nexus()
                nexus_1.fidelity_level         = 1
                nexus.surrogate_data.op.lf     = nexus_1
                nexus_1.surrogate_data         = surrogate_data

                nexus_1.optimization_problem   = problem
                nexus_1.vehicle_configurations = Vehicles_lf.setup(nexus_1)
                nexus_1.analyses  = Analyses_lf.setup(nexus_1.vehicle_configurations)
                nexus_1.missions  = Missions.setup(nexus_1.analyses,nexus.vehicle_configurations)
                nexus_1.procedure = Procedure.setup(nexus_1)
            if i == 2:
                print 'high fid'
                nexus_2           = Nexus()
                nexus_2.fidelity_level         = 2
                nexus.surrogate_data.op.hf     = nexus_2
                nexus_2.surrogate_data         = surrogate_data

                nexus_2.optimization_problem   = problem
                nexus_2.vehicle_configurations = Vehicles_hf.setup(nexus_2)
                nexus_2.analyses  = Analyses_hf.setup(nexus_2.vehicle_configurations)
                nexus_2.missions  = Missions.setup(nexus_2.analyses,nexus_2.vehicle_configurations)
                nexus_2.procedure = Procedure.setup(nexus_2)
        return nexus
    else:
        if fidelity_level > 0:
            print 'method route 2'
            if fidelity_method[0] == 1:
                nexus.vehicle_configurations= Vehicles_lf.setup(nexus)
                nexus.analyses              = Analyses_lf.setup(nexus.vehicle_configurations)
            elif fidelity_method[0] == 2:
                nexus.vehicle_configurations= Vehicles_hf.setup(nexus)
                nexus.analyses              = Analyses_hf.setup(nexus.vehicle_configurations)
            elif fidelity_method[0] == 0:
                nexus.vehicle_configurations= Vehicles_lf.setup(nexus)
                nexus.analyses              = Analyses_lf.setup(nexus.vehicle_configurations)
            else:
                raise Exception(pick_again)
                exit()
            nexus.missions              = Missions.setup(nexus.analyses,nexus.vehicle_configurations)
            nexus.procedure             = Procedure.setup(nexus)        

        # otherwise, business as usual
        else:
            print 'method route 3'
            nexus.vehicle_configurations= Vehicles_lf.setup(nexus)
            nexus.analyses              = Analyses_lf.setup(nexus.vehicle_configurations)
            nexus.missions              = Missions.setup(nexus.analyses,nexus.vehicle_configurations)
            nexus.procedure             = Procedure.setup(nexus)


    return nexus

def plot3D(X,y,inpstr,objstr,obj_choice=0):
    

    if np.shape(X)[0] > np.shape(X)[1]:
        x1  = X[:,0]
        x2  = X[:,1]
    else:
        x1  = X[0,:]
        x2  = X[1,:]

    y   = y[:,obj_choice]
    print x1
    print x2
    print y

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_surface(x1, x2, y, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.plot_wireframe(x1, x2, y, rstride=10, cstride=10)


    # Customize the z axis.
#    ax.zaxis.set_major_locator(LinearLocator(10))
#    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Labels
    ax.set_xlabel(inpstr[0])
    ax.set_ylabel(inpstr[1])
    ax.set_zlabel(objstr[0])

    # Add a color bar which maps values to colors.
#    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


    return


if __name__ == '__main__':
    main()


#    surr.sample_plan.lhc = np.array(\
#[[ 1.7625e+01,  5.7525e-01,  6.6250e-01,  4.7500e-01,  1.0500e+01, -8.5000e+00,  -4.7500e+00,  2.5000e-01, -1.2500e+00,  1.7500e+00],\
# [ 1.6675e+01,  2.1975e-01,  7.9750e-01,  1.2500e-01,  3.4500e+01,  4.5500e+01,   2.7500e+00, -2.5000e-01,  2.5000e-01,  4.2500e+00],\
# [ 3.3750e+00,  4.9625e-01,  4.8250e-01,  7.5000e-02,  1.6500e+01,  3.0500e+01,   3.7500e+00, -3.2500e+00, -2.2500e+00, -4.2500e+00],\
# [ 5.2750e+00,  4.5675e-01,  6.1750e-01,  9.7500e-01,  4.0500e+01,  1.8500e+01,   3.2500e+00,  4.7500e+00, -4.2500e+00,  2.2500e+00],\
# [ 1.2875e+01,  4.1725e-01,  4.3750e-01,  7.2500e-01,  4.6500e+01,  3.3500e+01,   7.5000e-01, -7.5000e-01,  4.7500e+00,  2.7500e+00],\
# [ 1.0025e+01,  9.3075e-01,  8.4250e-01,  8.7500e-01,  3.1500e+01,  4.2500e+01,  -2.7500e+00, -2.7500e+00,  7.5000e-01, -4.7500e+00],\
# [ 1.9525e+01,  7.3325e-01,  8.8750e-01,  6.7500e-01,  2.5500e+01,  2.4500e+01,   1.7500e+00, -1.7500e+00, -4.7500e+00,  3.2500e+00],\
# [ 1.5725e+01,  5.3575e-01,  2.5750e-01,  3.2500e-01,  1.9500e+01,  1.5500e+01,  -2.2500e+00, -3.7500e+00,  2.2500e+00,  3.7500e+00],\
# [ 1.8575e+01,  8.5175e-01,  7.5250e-01,  2.5000e-02,  2.2500e+01,  2.1500e+01,  -1.2500e+00, -1.2500e+00,  3.2500e+00, -1.7500e+00],\
# [ 6.2250e+00,  7.7275e-01,  9.7750e-01,  5.7500e-01,  5.2500e+01,  6.5000e+00,  -7.5000e-01,  3.2500e+00, -3.7500e+00, -2.7500e+00],\
# [ 1.0975e+01,  9.7025e-01,  3.9250e-01,  6.2500e-01,  7.5000e+00,  9.5000e+00,   2.5000e-01, -4.7500e+00, -1.7500e+00, -1.2500e+00],\
# [ 1.3825e+01,  2.5925e-01,  5.2750e-01,  1.7500e-01,  5.5500e+01,  3.5000e+00,  -1.7500e+00,  4.2500e+00,  4.2500e+00, -3.7500e+00],\
# [ 9.0750e+00,  3.7775e-01,  3.4750e-01,  4.2500e-01,  4.9500e+01,  3.9500e+01,  -4.2500e+00, -2.2500e+00, -3.2500e+00, -3.2500e+00],\
# [ 8.1250e+00,  6.9375e-01,  2.1250e-01,  5.2500e-01,  4.5000e+00,  3.6500e+01,   4.2500e+00,  1.7500e+00,  3.7500e+00, -2.5000e-01],\
# [ 2.4250e+00,  6.5425e-01,  9.3250e-01,  7.7500e-01,  1.3500e+01,  1.2500e+01,   4.7500e+00,  7.5000e-01, -2.5000e-01,  1.2500e+00],\
# [ 1.4775e+01,  8.1225e-01,  1.6750e-01,  2.7500e-01,  4.3500e+01, -2.5000e+00,   2.2500e+00,  2.7500e+00,  1.2500e+00, -7.5000e-01],\
# [ 1.1925e+01,  2.9875e-01,  3.0250e-01,  8.2500e-01,  2.8500e+01, -5.5000e+00,  -3.2500e+00,  3.7500e+00, -2.7500e+00, -2.2500e+00],\
# [ 7.1750e+00,  6.1475e-01,  1.2250e-01,  9.2500e-01,  5.8500e+01,  2.7500e+01,   1.2500e+00, -4.2500e+00, -7.5000e-01,  2.5000e-01],\
# [ 1.4750e+00,  3.3825e-01,  5.7250e-01,  2.2500e-01,  1.5000e+00,  4.8500e+01,  -2.5000e-01,  2.2500e+00,  1.7500e+00,  7.5000e-01],\
# [ 4.3250e+00,  8.9125e-01,  7.0750e-01,  3.7500e-01,  3.7500e+01,  5.0000e-01,  -3.7500e+00,  1.2500e+00,  2.7500e+00,  4.7500e+00]])
