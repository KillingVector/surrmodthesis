# cokrig matops.py
# A.A.Wachman

import numpy as np
from numpy.matlib import rand,zeros,ones,empty,eye
import scipy


class matrixops():

    def __init__(self):
        self.LnDesPsidXe= None
        self.PsidXe     = np.zeros((self.ne,self.ne), dtype=np.float)
        self.psidXe     = np.zeros((self.ne,1))
        self.oned       = np.ones((self.ne,1))  # must transpose
        self.onec       = np.ones((self.nc,1))
        self.mud        = None
        self.Ud         = None
        self.SigmaSqrd  = None
        self.Lambdad    = None
        self.NegLnLiked = None
        print np.shape(self.Xe)
        print np.shape(self.Xc)
        print 'matrix ops init complete'
#        quit()




        print 'matrixops init complete'

#####################################################################
#####################   UPDOOT SECTION  #############################
#####################################################################

        # update distances
    def updateData(self):
        #   Difference Vector
        self.updateDifferenceVector()
        # cheap
        self.distanceXc = np.zeros((self.nc,self.nc, self.kc))
        for i in range( self.nc ):
            for j in range(i+1,self.nc):
                self.distanceXc[i][j]=np.abs((self.Xc[i,:]-self.Xc[j,:]))

#        print self.distanceXc
        #   expensive\
#        print self.ne
#        print np.shape(self.Xe)
        self.distanceXe = np.zeros((self.ne,self.ne, self.ke))
        for i in range( self.ne ):
            for j in range(i+1,self.ne):
                self.distanceXe[i][j]=np.abs((self.Xe[i,:]-self.Xe[j,:]))
        #   === CHEAP-EXP DISTANCES
        self.distanceXcXe = np.zeros((self.nc,self.ne, self.kc))
        for i in range( self.nc ):
            for j in range(self.ne):
                self.distanceXcXe[i][j]=np.abs((self.Xc[i,:]-self.Xe[j,:]))
#        quit()


    #   this creates the distance vector d
    def updateDifferenceVector(self):
        """
            This function combs through the data and vectors that 
            are present in both Xe and Xc and calculates d - the
            difference via a regression parameter, rho, which is a
            parameter for which we optimize
        """
#        print np.shape(self.Xe)
#        print np.shape(self.Xc)

#        print self.Xc
#        print self.Xe

        # storage
        xe = []
        ye = []
        xc = []
        yc = []

        xd = []
        d  = []

        fillxc=[]
        fillyc=[]

        ct1 = 0
        ct2 = 0
        ct3 = 0
        for j in range(0,np.shape(self.Xc)[0]):
            entry = self.Xc[j,:]
            present = False
            for i in range(0,np.shape(self.Xe)[0]):
                test    = self.Xe[i,:]
                dist    = np.absolute(np.linalg.norm(entry - test))
                # values are normalised in here, so max is 1
                tol1 = 1.0e-4
                tol2 = 1.0e-6 # tolerance for distances
                # using normalised distance to iD if values are the same
#                if dist < tol2:
                if np.allclose(entry,test):
                    if np.absolute(test[0] - entry[0]) < tol1:
                        xe.append(test.tolist())
                        ye.append(self.ye[i,0].tolist())
                        xc.append(entry.tolist())
                        yc.append(self.yc[j,0].tolist())
                        xd.append(entry.tolist())
                        d.append(self.ye[i,0].tolist() - self.rho * self.yc[j,0].tolist())
                        present = True


            if not present:
                fillxc.append(entry.tolist())
                fillyc.append(self.yc[j,0].tolist())


        xc  = np.atleast_2d(np.array(xc))
        fillxc = np.atleast_2d(np.array(fillxc))
        yc  = np.atleast_2d(np.array(yc)).T
        fillyc = np.atleast_2d(np.array(fillyc)).T

        xc  = np.concatenate((xc,fillxc),axis=0)
        yc  = np.concatenate((yc,fillyc),axis=0)
#        print yc
#        quit()

        # reallocate back to original arrays
        self.Xe = np.array(xe)
        self.ye = np.atleast_2d(np.array(ye)).T
        self.Xc = xc
        self.yc = yc
        self.Xd = np.array(xd)
#        self.d  = np.abs(np.atleast_2d(np.array(d)).T) # ? yes? or should it be negative
        self.d  = np.atleast_2d(np.array(d)).T
        self.y  = np.concatenate((self.yc,self.ye), axis=0)
        #   atleast_2d keeps it so that it is transposable
#        print np.shape(self.Xe)
#        print self.Xe
#        print np.shape(self.Xc)
#        print np.shape(self.Xd)
#        quit()
#        self.nc = self.Xc.shape[0]   # rows - num samps
#        self.kc = self.Xc.shape[1]   # cols - num vars
#        self.ne = self.Xe.shape[0]   # same shit but expensive
#        self.ke = self.Xe.shape[1]   # 
        # To account for items not used, rescale these.
        # hours on this problem and the bloody answer is four lines of reassignment. Ugh.
        



    def updatePsi(self):
        """
            Updates all the Psi values
        """ 

        self.updatePsicXc()
        self.updatePsidXe()
        self.updatePsicXe()
        self.updatePsicXcXe() 
#        print 'All Psi updated'  



        return 'I have no mouth and I must scream'

    #   Psi_c(Xc,Xc)
    def updatePsicXc(self):
        # unpack data
        Xc      = self.Xc
        yc      = self.yc
        nc      = self.nc
        kc      = self.kc
        p       = self.p

        thetac  = self.thetac   # c in candidates will replace this
        rho     = self.rho
#        one     = np.ones((nc,1))

        PsicXc  = np.zeros((nc,nc))
        self.PsicXc = PsicXc        
        for i in range(0,nc):
            for j in range(i+1,nc):
#                distc       = np.power(np.abs(Xc[i,:] - Xc[j,:]), p)
                distc       =np.power(np.abs(self.distanceXc[i,j]),p)
                PsicXc[i,j] = np.exp(-np.sum(np.multiply(thetac,distc)))
        self.PsicXc = PsicXc
        self.PsicXc = PsicXc + PsicXc.T + np.eye(nc) + np.multiply(np.eye(nc), np.spacing(1))

        try:
            self.UXc     = np.linalg.cholesky(self.PsicXc)
        except Exception as err:
            print 'PsicXc : ' + str(err)
            self.UXc = 1.e2
#        self.UXc     = self.UXc.T

    #   Psi_d(Xe,Xe)
    def updatePsidXe(self):
        # unpack data
        Xe      = self.Xe
        ye      = self.ye
        yc      = self.yc
        ne      = self.ne
        ke      = self.ke
        p       = self.p

        thetad  = self.thetad   # c in candidates will replace this
        rho     = self.rho
        one     = np.ones((ne,1))
        PsidXe  = np.ones((self.ne,self.ne), dtype=np.float64)

        # Now we do the updates for PsidXe
        for i in range(0,ne):
            for j in range(i+1,ne):
                distd       =np.power(np.abs(self.distanceXe[i,j]),p)
#                distd       = np.power(np.abs(Xe[i,:] - Xe[j,:]), p)
                PsidXe[i,j] = np.exp(-np.sum(np.multiply(thetad,distd)))

        # add upper and lower halves and diagonal of ones 
        # also add small number to reduce ill-conditioning
                        # avoid non pos det?
        psihold = PsidXe + PsidXe.T + np.eye(ne) + np.multiply(np.eye(ne), np.spacing(1))
        # oh just resetting things <3
        self.PsidXe  = psihold
        PsidXe       = psihold
        # I HERD U LIKE CHOLESKY FACTORISATIONS
        try:
            Ud      = np.linalg.cholesky(self.PsidXe)
            self.Ud = Ud.T
        except Exception as err:
            print 'PsidXe : ' + str(err)
            self.Ud = 1.e4

    #   Psi_c(Xe,Xe)
    def updatePsicXe(self):
        # unpack data
        Xe      = self.Xe
        ye      = self.ye
        yc      = self.yc
        ne      = self.ne
        ke      = self.ke
        p       = self.p

        thetac  = self.thetac   # c in candidates will replace this
        rho     = self.rho
        one     = np.ones((ne,1))

        self.PsicXe = np.zeros((ne,ne))
        PsicXe      = self.PsicXe
        for i in range(0,ne):
            for j in range(i+1,ne):
                distce      =np.power(np.abs(self.distanceXe[i,j]),p)
#                distce      = np.power(np.abs(Xe[i,:] - Xe[j,:]), p)
                PsicXe[i,j] = np.exp(-np.sum(np.multiply(thetac,distce)))
        self.PsicXe = PsicXe
        self.PsicXe = PsicXe + PsicXe.T + np.eye(ne) + np.multiply(np.eye(ne), np.spacing(1))
        try:
            self.UPsicXe= np.linalg.cholesky(self.PsicXe)
        except Exception as err:
            print 'PsicXe : ' + str(err)
            self.UPsiXe = 1.e4

    #   Psi_c(Xc,Xe) and Psi_c(Xe,Xc)
    def updatePsicXcXe(self):
        Xe      = self.Xe
        Xc      = self.Xc
        thetac  = self.thetac
        nc      = self.nc
        ne      = self.ne
        p       = self.p

        self.PsicXcXe   = np.zeros((nc,ne))
        PsicXcXe        = self.PsicXcXe
        for i in range(0,nc):
            for j in range(0,ne):
                distcce   =np.power(np.abs(self.distanceXcXe[i,j]),p)
#                distcce      =np.power(np.abs(Xc[i,:] - Xe[j,:]), p)
                PsicXcXe[i,j]=np.exp(-np.sum(np.multiply(thetac,distcce)))
        self.PsicXcXe   = PsicXcXe
        eyene           = np.eye(ne)
        zeronc          = np.zeros((nc-ne,ne))
        zeroeye         = np.concatenate((zeronc,eyene), axis=0)

        self.PsicXcXe   = PsicXcXe + np.multiply(zeroeye,np.spacing(1))
        self.PsicXeXc   = self.PsicXcXe.T

            


#####################################################################
######################    LIKELIHOOD    #############################
#####################################################################

    #   CHEAP LIKELIHOOD
    def neglikelihoodc(self): # based on likelihoodc.m
        Xc      = self.Xc
        yc      = self.yc
        nc      = self.nc
        kc      = self.kc
        p       = self.p

        thetac  = self.thetac   # c in candidates will replace this
        rho     = self.rho
        onec     = np.ones((nc,1))
        # first we do cheap stuff
        PsicXc  = np.zeros((nc,nc))
        self.PsicXc = PsicXc        
        for i in range(0,nc):
            for j in range(i+1,nc):
#                distc       = np.power(np.abs(Xc[i,:] - Xc[j,:]), p)
                distc       =np.power(np.abs(self.distanceXc[i,j]),p)
                PsicXc[i,j] = np.exp(-np.sum(np.multiply(thetac,distc)))
        self.PsicXc = PsicXc
        self.PsicXc = PsicXc + PsicXc.T + np.eye(nc) + np.multiply(np.eye(nc), np.spacing(1))

        try:
            self.UXc     = np.linalg.cholesky(self.PsicXc)
#            self.UXc     = self.UXc.T
        except Exception as err:
            print 'neglikelihood C : ' + str(err)
            self.NegLnLikec = 1.e2
            return self.NegLnLikec

        UXc = self.UXc
        self.LnDetPsicXc    = 2 * np.sum(np.log(np.abs(np.diag(UXc))))
        # num: One.T * (UXc * (UXc.T * yc)) - One, yc: col vec
        UTy     = np.linalg.solve(UXc.T,yc)
        UUTy    = np.linalg.solve(UXc,UTy)
        muc_num = onec.T.dot(UUTy)
        # den: One.T * (UXc * (UXc.T * One))   - One: col vec
        UT1     = np.linalg.solve(UXc.T,onec)
        UUT1    = np.linalg.solve(UXc,UT1)
        muc_den = onec.T.dot(UUT1)
        self.muc= muc_num / muc_den
#        print 'made mud : ' + str(self.mud)

        # sgc2  = (yc - 1*muc).T * inv(PsicXc) * (yc - 1*muc) / nc
        # num: (yc - 1*muc) * (UXc * (UXc.T * (yc - 1*muc)))
        y1mu    = yc - onec*self.muc
        Uy1mu   = np.linalg.solve(UXc.T,y1mu)
        UUy1mu  = np.linalg.solve(UXc,Uy1mu)
        sig_num = y1mu.T.dot(UUy1mu)
        sig2c   = sig_num / nc
        self.SigmaSqrc = sig2c

        self.NegLnLikec=-1.*(-(self.nc/2.)*np.log(self.SigmaSqrc) - 0.5*self.LnDetPsicXc)

        return self.NegLnLikec


    #   DIFF LIKELIHOOD
    def neglikelihoodd(self): # based on likelihoodd.m
        Xe      = self.Xe
        ye      = self.ye
        yc      = self.yc
        ne      = self.ne
        ke      = self.ke
        p       = self.p
        
        thetad  = self.thetad   # c in candidates will replace this
        rho     = self.rho
        one     = np.ones((ne,1))

        PsidXe  = self.PsidXe


        #   build upper half of correlation matrix
        for i in range(0,ne):
            for j in range(i+1,ne):
                distdxe     =np.power(np.abs(self.distanceXe[i,j]),p)
#                distdxe     = np.power(np.abs(Xe[i,:] - Xe[j,:]), p)
                PsidXe[i,j] = np.exp(-np.sum(np.multiply(thetad,distdxe)))

        # add upper and lower halves and diagonal of ones 
        # also add small number to reduce ill-conditioning
                        # avoid non pos det?
        psihold = PsidXe + PsidXe.T + np.eye(ne) + np.multiply(np.eye(ne), np.spacing(1))
        # oh just resetting things <3
        self.PsidXe  = psihold
        PsidXe       = psihold
        # I HERD U LIKE CHOLESKY FACTORISATIONS
        try:
            Ud      = np.linalg.cholesky(self.PsidXe)
            self.Ud = Ud
        except Exception as err:
            print 'neglikelihood D : ' + str(err)
            self.NegLnLiked = 1.e4 # return large number to divert
            return self.NegLnLiked
#        print 'created Ud'
        # I don't understand maths enough to know why this works:
        # sum lns of diagonal of Ud to find ln(det(PsidXe))
        LnDetPsidXe     = 2* np.sum(np.log(np.abs(np.diag(self.Ud))))
        self.LnDetPsidXe= LnDetPsidXe

#        print 'made log determinant : ' + str(self.LnDetPsidXe)
        d       = self.d  # 'normal d' needs to be col vec

        # mud   = (1.T * inv(PsidXe) * d) / (1.T * inv(PsidXe) * 1)
        # inv(PsidXe) = U\(U.T\d)
        UTd     = np.linalg.solve(Ud.T,d)
        UUTd    = np.linalg.solve(Ud,UTd)
        mud_num = self.oned.T.dot(UUTd)
        # inv(PsidXe) = U\(U.T\1)
        UT1     = np.linalg.solve(Ud.T,self.oned)
        UUT1    = np.linalg.solve(Ud,UT1)
        mud_den = self.oned.T.dot(UUT1)
        # checked.  num always neg, den always pos?
        self.mud= mud_num / mud_den
#        print 'made mud : ' + str(self.mud)

        # sgd2  = (d - 1*mud).T * inv(PsidXe) * (d - 1*mud) / ne
        d1mu    = d - self.oned*self.mud
        Ud1mu   = np.linalg.solve(Ud.T,d1mu)
        UUd1mu  = np.linalg.solve(Ud,Ud1mu)
        sig_num = d1mu.T.dot(UUd1mu)
        sig2d   = sig_num / self.ne
        self.SigmaSqrd = sig2d
#        print 'made sig2d : ' + str(self.SigmaSqrd)

        NegLnLiked = -1 * (-(self.ne/2) * np.log(self.SigmaSqrd) - 0.5*self.LnDetPsidXe)
#        print 'neglikelihoodd : ' + str(NegLnLiked)
        self.NegLnLiked = NegLnLiked

        return self.NegLnLiked


#####################################################################
######################   BUILD  THE  C  #############################
#####################################################################


    def buildC(self):
        """
            Build C.
                (   A   ,   B   )
                (   C   ,   D   )

            A   = sig2c*PsicXc
            B   = sig2c*PsicXcXe
            C   = sig2c*PsicXeXc
            D   = rho2*sig2c*PsicXe*sig2d*PsidXe 
        """

        sig2c   = self.SigmaSqrc
        sig2d   = self.SigmaSqrd
        rho     = self.rho
        nc      = self.nc
        ne      = self.ne
        one     = np.ones((nc+ne,1))
        y       = self.y

        PsicXc  = self.PsicXc
        PsicXcXe= self.PsicXcXe
        PsicXeXc= PsicXcXe.T
        PsicXe  = self.PsicXe
        PsidXe  = self.PsidXe


        print sig2c
        print PsicXc 
        print sig2d
        print PsidXe

        A   = sig2c * PsicXc
        B   = rho * sig2c * PsicXcXe
        C   = rho * sig2c * PsicXeXc
        D1  = rho**2 * sig2c * PsicXe
        D2  = sig2d * PsidXe
        D   = D1 + D2
        # none of these are zero
        AB  = np.concatenate((A,B), axis=1)
        CD  = np.concatenate((C,D), axis=1)

        print 'A is pdef: ' + str(self.is_pos_def(A)) #+ '\n' + str(A)
        print 'B is pdef: ' + 'not square' #+ '\n' + str(B)
        print 'C is pdef: ' + 'not square' #+ '\n' + str(C)
        print 'D1 is pdef: ' + str(self.is_pos_def(D1)) #+ '\n'+ str(D1)
        print 'D2 is pdef: ' + str(self.is_pos_def(D2)) #+ '\n'+ str(D2)
        print 'D is pdef: ' + str(self.is_pos_def(D)) #+ '\n' + str(D)

        Cmat= np.concatenate((AB,CD), axis=0)

        self.C  = Cmat
        cmatstat=self.is_pos_def(self.C)
        print 'C_matrix : pos def : ' + str(cmatstat)
#        print self.C
        invert  = False
        self.UC = None
        try:
            self.UC = np.linalg.cholesky(Cmat)  
            # even if matrix is technically pos def, this may not work
        except Exception as err:
            print err
            print 'Attempting pseudo-inverse'
            invert  = True
        if invert:  # if C was inverted via pseudo invert
            try:
                print 'Using pinv: Cmat technically pos def: ' + str(cmatstat)
                invC = np.linalg.pinv(Cmat)
                self.invC = invC
            except Exception as err:
                print err
                print 'Matrix is non-invertible'
            cy  = invC.dot(self.y)
            num = one.T.dot(cy)
            c1  = invC.dot(one)
            den = one.T.dot(c1)

            self.mu = num / den
            dmu = y - one.dot(self.mu)
            self.SigmaSqr = (one.T.dot(self.invC.dot(dmu)))/(self.nc+self.ne)

        if not invert:  # C can be cholesky'd
            ucy  = np.linalg.solve(self.UC.T,y)
            uucy = np.linalg.solve(self.UC,ucy)
            mu_num=one.T.dot(uucy)
            print 'numerator : ' + str(mu_num)
            uc1  = np.linalg.solve(self.UC.T,one)
            uuc1 = np.linalg.solve(self.UC,uc1)
            mu_den=one.T.dot(uuc1)
            self.mu = mu_num / mu_den

            dmu  = y - one.dot(self.mu)
            ucd  = np.linalg.solve(self.UC.T,dmu)
            uucd = np.linalg.solve(self.UC,ucd)
            self.SigmaSqr = (dmu.T.dot(uucd)) / (self.nc + self.ne)

        print '=========='
        print 'mu       : ' + str(self.mu)
        print 'Sig2     : ' + str(self.SigmaSqr)
        print 'mu_d     : ' + str(self.mud)
        print 'Sig2d    : ' + str(self.SigmaSqrd)
        print 'mu_c     : ' + str(self.muc)
        print 'Sig2c    : ' + str(self.SigmaSqrc)




        return












    def is_pos_def(self,x):
        return np.all(np.linalg.eigvals(x) > 0)            

