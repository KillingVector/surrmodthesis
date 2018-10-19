#====================================================================
#   Imports
#====================================================================

import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd    
import time
from SUAVE.Core import Data


#====================================================================
#   Make Components
#====================================================================

def generate_vehicle(nexus):

    plot = False

    if hasattr(nexus,'Surrogate_Data'):
        the_vec     = nexus.Surrogate_Data.vec
    elif hasattr(nexus,'surrogate_data'):
        the_vec     = nexus.surrogate_data.vec
    elif hasattr(nexus,'surrogate'):
        the_vec     = nexus.surrogate.vec
    else:
        Exception("Please name Surrogate_Data() structure as \'surrogate_data\' in Optimize.py")

    material    = the_vec.material
    aerofoil    = the_vec.aerofoil
    payload     = the_vec.payload

    if the_vec.make_vec == None:
        vec = the_vec.init_vec
    else:
        vec = the_vec.make_vec

    pts         = get_points(vec)
    geom        = get_geometry(vec, pts, aerofoil) # area vectors
    aero_geom, A_ref, span_vals, total_mass=get_aero_geometry(vec, material, pts, aerofoil, geom, payload,plot)

    s           = vec.span              # span
    rootChord   = vec.root_chord        # root chord
    rcp         = np.array(vec.rcp)     # root chord percent (taper)
    psl         = np.array(vec.psl)     # percentage span location
    sqc         = np.array(vec.sqc)     # sweeps leading edge
    do          = np.array(vec.do)      # dihedral outboard
    tw          = np.array(vec.tw)      # twizzle sticks
    ttc         = np.array(vec.ttc)


    results     = s, rootChord, rcp, psl, sqc, do, tw, ttc, aero_geom, A_ref, span_vals, total_mass
#    print 'sqc : ' + str(sqc)
    return  results


    #   Get points
def get_points(vec):

    # define x,y,z direction
    #   z: pos/up (alt-wise)
    #   y: spanwise
    #   x: chordwise    
    
    #   Unpack
    s           = vec.span              # span
    rootChord   = vec.root_chord        # root chord
    rcp         = np.array(vec.rcp)     # root chord percent (taper)
    psl         = np.array(vec.psl)     # percentage span location
    sqc         = np.array(vec.sqc)     # sweeps leading edge
    do          = np.array(vec.do)      # dihedral outboard
    tw          = np.array(vec.tw)      # twizzle sticks
    ttc         = np.array(vec.ttc)
#    print 'rcp :' + str(rcp)
#    print 'do  :' + str(do)
    dim                 = len(psl+1)        # dim of point vecs 
    
    leading_edge_points = np.zeros((dim+1,3))
    trailing_edge_points= np.zeros((dim+1,3))


    #   initialize variables
    le_chordwise        = np.zeros((dim+1))
    le_z                = np.zeros((dim+1))

    #   get spanwise pos
    le_spanwise = np.concatenate((np.array(s/2*psl),[s/2]),axis=0)

    #   calculate non-twist point positions
    for i in range(1,dim+1):
        s_i             = le_spanwise[i]-le_spanwise[i-1]
        qcpi            = s_i * np.tan(np.deg2rad(sqc[i-1])) + le_chordwise[i-1]
        le_chordwise[i] = qcpi + rcp[i] * rootChord * 0.25
#        le_chordwise[i] = s_i * np.tan(np.deg2rad(sle[i-1]))   \
#                          + le_chordwise[i-1]
        edge2D_i        = ((le_spanwise[i]-le_spanwise[i-1])**2 + \
                          (le_chordwise[i]-le_chordwise[i-1])**2)**.5
        le_z[i]         = edge2D_i * np.tan(np.deg2rad(do[i-1]))   \
                          + le_z[i-1]

    section_chords      = rootChord * rcp
    te_chordwise        = le_chordwise + section_chords

    # get non-vars (don't worry about twist, it doesn't change areas)
    te_spanwise         = le_spanwise
    te_z                = le_z
    #   create list of tuples
    for i in range(0,dim+1):
        leading_edge_points[i,:] = [le_chordwise[i], le_spanwise[i], le_z[i]]
        trailing_edge_points[i,:]= [te_chordwise[i], le_spanwise[i], le_z[i]]
#    print 'LE and TE  points'
#    print leading_edge_points
#    print trailing_edge_points
    all_points      = [leading_edge_points, trailing_edge_points]

    return all_points


    #   calculate geometry
def get_geometry(vec, all_points, aerofoil):
    
    # take in leading and trailing edge points, vec
    # calculate areas, centroids of mass and aero

    #   get points for wetted and planform areas
    le_pts          = all_points[0]
    te_pts          = all_points[1]
    #   this rubbish is to make an new copy, rather than shallow
    le_pts_pln      = np.array(list(le_pts))
    te_pts_pln      = np.array(list(te_pts))
    le_pts_pln[:,2] = 0
    te_pts_pln[:,2] = 0

    # set and declare lists
    le_pts.tolist()
    te_pts.tolist()
    le_pts_pln.tolist()
    te_pts_pln.tolist()

    vec_areas_wetted    = []
    vec_areas_planform  = []


    for i in range(0,len(le_pts)-1):
        # this will be wrong for aerofoil, but we'll come back to it
        wet_list    = [le_pts[i],le_pts[i+1],te_pts[i+1],te_pts[i]]
        pln_list    = [le_pts_pln[i],le_pts_pln[i+1],te_pts_pln[i+1],te_pts_pln[i]]
        vec_areas_wetted.append(area(wet_list))
        vec_areas_planform.append(area(pln_list))

    # Pack!
    geom    = vec_areas_wetted, vec_areas_planform
    return geom


    #   calculate aero geometry
def get_aero_geometry(vec, material, all_points, aerofoil, geom, payload,plot = False):

    # unpack
    rootChord   = vec.root_chord
    span        = vec.span
    rcp         = vec.rcp
    ttc         = np.array(vec.ttc)
    sqc         = vec.sqc
    dim         = len(rcp) - 1

    rho         = material.rho
    Exx         = material.Exx
    Eyy         = material.Eyy
    G           = material.G
    poisson     = material.poisson

    vec_areas_wetted, vec_areas_planform    = geom

    [le_xyz, te_xyz]    = all_points
    le_x    = le_xyz[:,0]
    te_x    = te_xyz[:,0]
    le_y    = le_xyz[:,1]

    c_bar = np.zeros([dim])
    c_bar_i = np.zeros([dim])

    #   calculate reference areas
    A_ref_wet_half  = np.sum(vec_areas_wetted)
    A_ref_wet       = A_ref_wet_half * 2
    A_ref_pln_half  = np.sum(vec_areas_planform)
    A_ref_pln       = A_ref_pln_half * 2   

#    print 'vector of areas : ' + str(vec_areas_wetted)
#    print 'ref wet areas   : ' + str(A_ref_wet)

    #   calculate c_bar
    for j in range(0,dim):
        coef     = 2./3. * rootChord * rcp[j]
        segTaper = rcp[j+1] / rcp[j]
        frac     = (segTaper**2 + segTaper + 1.)/(segTaper + 1.)
        c_bar[j] =  coef * frac
        c_bar_i[j]= vec_areas_wetted[j] * c_bar[j]
    
    c_bar_num = np.sum(c_bar_i)
    c_bar= c_bar_num / A_ref_wet_half

    #   calculate aerofoil mass and aero centroids
    if aerofoil == None:
        com = 0					# wing skin com
        coa = 0					# wing aero centroid
        # create useful geometries, average com is ~375% c
        c_quart	= le_x + 0.25*(te_x - le_x)	# quarter chord
        c_avcom	= le_x + 0.375*(te_x - le_x)	# .375 for com

        thicc       = np.average(vec.ttc)

        skin_mass   = A_ref_wet * 2 * rho * 0.0005

        mthick_i    = 0.375
#        x           = [0., c_avcom, 1.]
#        y           = [0., 
#        aerof_perimeter 

    else: # given aerofoil data
        aerof_pts   = aerofoil.pts
        shp         = np.shape(aerof_pts)
        if shp[0] > shp[1]:
            x       = aerof_pts[:,0]
            y       = aerof_pts[:,1]
        else:
            x       = aerof_pts[0,:]
            y       = aerof_pts[1,:]
        #   get weighted vector of x using y disp
        quart_i = np.sum(np.multiply(x,y))/np.sum(y)
        com_i   = vec.aerofoil.coa
        c_quart	= le_x + quart_i*(te_x - le_x)	# quarter chord
        c_avcom	= le_x + com_i*(te_x - le_x)
        
        mthick_i= np.argmax(y)
        aeroc_i = np.where(x==0.25)[0]
#        print y[aeroc_i]
        thicc   = (y[mthick_i] - y[len(y) - mthick_i]) *(te_x - le_x)


        aerof_perimeter = perimeter(x,y) * 0.005 # perim * skinthckns
#        print aerof_perimeter
        
    coa_aerofoil = 0.
    com_aerofoil = 0.
    # calculate mass centre and aero centre for whole craft
    for i in range(0,dim):
        coa_aerofoil = coa_aerofoil + ((c_quart[i]+c_quart[i+1])/2*vec_areas_planform[i])
        com_aerofoil = com_aerofoil + ((c_avcom[i]+c_avcom[i+1])*(vec_areas_wetted[i]*rho*0.0005))

    # get spar mass
    spar_mass   = size_spar(vec, material, payload, all_points, geom, thicc)
    spar_loc    = np.sum(np.multiply(vec.rcp,c_avcom)) / np.sum(vec.rcp)

    total_mass  = skin_mass + spar_mass     #   ADD SPAR MASS

    coa         = coa_aerofoil / A_ref_pln
    com_aero    = com_aerofoil/(A_ref_wet*2*rho*0.0005)
    com_wing    = (skin_mass*com_aero + spar_loc*spar_mass)/total_mass

    #   now add payload
    com         = coa - payload.static_margin*rootChord
    payload_com = ( com * total_mass - skin_mass * com_aero ) / payload.payload_mass
#    print 'force stab:payload com ref : ' + str(payload_com) + ' m'
    #   update total mass
#    print 'spar mass : ' + str(spar_mass)
#    print 'skin mass : ' + str(skin_mass)


#    print 'total mass: ' + str(total_mass)
    
    # last rec vals
    AR          = span**2 / A_ref_pln
    wing_taper  = rcp[-1]
    wing_sweep  = np.arctan(le_y[-1]/le_x[-1])
    wing_sweep_le  = 90-np.rad2deg(wing_sweep)
    wing_sweep_qc  = np.sum(np.multiply(vec_areas_planform,sqc)) / np.sum(vec_areas_planform)


    # Force stability, no payload
    com         = coa - payload.static_margin*rootChord
    total_mass  = total_mass + payload.payload_mass

    # Pack
    aero_geom   = c_bar, coa, com
    A_ref       = A_ref_wet, A_ref_pln
    span_vals   = wing_taper, wing_sweep_qc


    if plot:
        plot_bwb(vec, all_points, geom, [c_quart,c_avcom])

    return aero_geom, A_ref, span_vals, total_mass




def size_spar(vec, material, payload, all_points, geom, thicc):
        #   Calculate spar: Burton and Hoburg 2017
    """
        GRAV USED IN HERE, GOING TO NEED TO GET IT FROM SOMEWHERE
        
        GRAV = 9.81    
        
    """
    grav = 9.81

    # unpacks
    [le_xyz, te_xyz]    = all_points
    lex     = le_xyz[:,0]
    tex     = te_xyz[:,0]
    ley     = le_xyz[:,1]
    tey     = te_xyz[:,1]
    lez     = le_xyz[:,2]
    tez     = te_xyz[:,2]

    rho         = material.rho
    sig_c       = material.sig_c

    vec_areas_wetted, vec_areas_planform    = geom
    A_ref_wet_half  = np.sum(vec_areas_wetted)
    A_ref_wet       = A_ref_wet_half * 2
    A_ref_pln_half  = np.sum(vec_areas_planform)
    A_ref_pln       = A_ref_pln_half * 2 

    #   INIT VALUES
    # init w, t and h, assume max blocks
    w   = 0.3 * (tex - lex)
    t   = thicc/2 * (tex - lex)

    converged   = False
    tol         = 1.e-6 * sig_c
    count       = 0
#    print thicc

    #   Converge bar stress with ultimate stress    
    while not converged:

        h   = thicc/2 * (tex - lex) - t/2
        I   = w * t**3 /6 + 2*w*t*(h/2 + t/2)**2
        I   = np.average(I)

        cs  = np.multiply(w,t)
        vol = np.average(cs) * perimeter(lex,ley,lez)
        spar_mass=vol * rho
        skin_mass=A_ref_wet * 2 * rho * 0.0005
        mass    = spar_mass + skin_mass + payload.payload_mass
        L_gust  = 2 * mass * grav
        M       = L_gust * vec.span

        check   = np.absolute(M*thicc/I - 1.5*sig_c)
        if check <= tol:
            converged = True
        elif check >= tol and check <=1.e5*tol:
            if M*thicc/I < 1.5 * sig_c:
                w = w - w * 0.005
                t = t - t * 0.005
            else:
                w = w + w * 0.001
                t = t + t * 0.001
        else:
            if M*thicc/I < 1.5 * sig_c:
                w = w - w * 0.1
                t = t - t * 0.1
            else:
                w = w + w * 0.005
                t = t + t * 0.005
        # get booted out if you oscillate
        # this is a rough calc
        count = count + 1
        if count > 50:
            converged = True

    

    return spar_mass
    

    



#====================================================================
#   Auxiliary functions
#====================================================================

#unit normal vector of plane defined by points a, b, and c
def unit_normal(a, b, c):
    det = np.linalg.det
    x = det([[1,a[1],a[2]],
             [1,b[1],b[2]],
             [1,c[1],c[2]]])
    y = det([[a[0],1,a[2]],
             [b[0],1,b[2]],
             [c[0],1,c[2]]])
    z = det([[a[0],a[1],1],
             [b[0],b[1],1],
             [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return (x/magnitude, y/magnitude, z/magnitude)

#area of polygon poly
def area(poly):
    if len(poly) < 3: # not a plane - no area
        return 0

    total = [0, 0, 0]
    for i in range(len(poly)):
        vi1 = poly[i]
        if i is len(poly)-1:
            vi2 = poly[0]
        else:
            vi2 = poly[i+1]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)

def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length


def perimeter(x_vec,y_vec,z_vec = []):

    perimeter   = 0.
    if z_vec == []:
        z_vec   = np.zeros(len(x_vec))

    for i in range(0,len(x_vec)):
        x   = x_vec[i]
        y   = y_vec[i]
        z   = z_vec[i]
        l   = np.sqrt(x**2 + y**2 + z)
        perimeter = perimeter + l       
    

    return perimeter


def plot_bwb(vec, all_points, geom, aero):

    #   Unpack
    [le_xyz, te_xyz]    = all_points
    lex    = le_xyz[:,0]
    ley    = le_xyz[:,1]
    lez    = le_xyz[:,2]
    tex    = te_xyz[:,0]
    tey    = te_xyz[:,1]
    tez    = te_xyz[:,2]

    c_quart, c_avcom = aero

    #   Make Mesh
    # at some point, but not really necessary

    #   LE-TE or Surface
    fig = plt.figure()
    ax  = fig.add_subplot(111,projection='3d')

    ax.plot(lex,ley,lez,'k-')
    ax.plot(lex,-ley,lez,'k-')
    ax.plot(tex,tey,tez, 'k-')
    ax.plot(tex,-tey,tez,'k-')
    ax.plot(c_quart,ley,lez,c='r',linestyle='dashed')
    ax.plot(c_avcom,ley,lez,'g-')
    ax.plot(c_quart,-ley,lez,c='r',linestyle='dashed')
    ax.plot(c_avcom,-ley,lez,'g-')

    # need the final com

    #   make border box
    mx     = ley.max()
    border  =np.array([[0., mx, mx], [0.,-mx, mx], [0., mx,-mx], [0.,-mx, mx], [mx, mx, mx], [mx,-mx, mx], [mx, mx,-mx], [mx,-mx, mx]])
    for pt in border:
        ax.plot([pt[0]], [pt[1]], [pt[2]], 'w')
    ax.axis('equal')


    plt.show()

    

    




    return




#====================================================================
#   Testing area for testing things
#====================================================================



if __name__ == '__main__':


    # test structure
    nexus               = Data()
    nexus.Surrogate_Data= Data()
    nexus.Surrogate_Data.vec= Data()
    
    init_vec                 = Data()
    init_vec.span            = 6.
    init_vec.root_chord      = 2.5

#    init_vec.psl             = np.array([0., .40, .95])
#    init_vec.sle             = np.array([50., -25., 70.])
#    init_vec.rcp             = np.array([1., .6, 1.4, .2]) [-1] is tip taper
#    init_vec.do              = np.array([5., -15., 60.])
#    init_vec.tw              = np.array([-5., 1.])  just root and tip
#    init_vec.ttc             = np.array([0.2, 0.08])

    init_vec.psl             = np.array([0., .30])
    init_vec.sqc             = np.array([45., 25.])
    init_vec.rcp             = np.array([1., .5, .3]) #[-1] is tip taper
    init_vec.do              = np.array([5., 1.])
    init_vec.tw              = np.array([-5., 1.]) # just root and tip
    init_vec.ttc             = np.array([0.2, 0.08]) # thickness

    nexus.Surrogate_Data.vec.aerofoil        = None  # None or load aerofoil control pts
#    init_vec.aerofoil        = Data()
#    init_vec.aerofoil.pts    = pd.read_csv('naca2412.csv',sep=',',header=None).values
#    init_vec.aerofoil.coa    = .25

#    print init_vec.aerofoil
    

    init_vec.num_seg         = len(init_vec.psl)
    payload         = Data()
    payload.payload_mass    = 10.
    payload.static_margin   = .1
    material        = Data()
    material.Exx    = 70.0e9
    material.Eyy    = 70.0e9
    material.G      = 26e9
    material.rho    = 2700.
    material.poisson= 0.35
    material.sig_c  = 200.0e6
    material.sig_t  = 300.0e6
    opt_names       = ['span','root_chord']

    nexus.Surrogate_Data.vec.init_vec     = init_vec
    nexus.Surrogate_Data.vec.make_vec     = None
    nexus.Surrogate_Data.vec.material     = material
    nexus.Surrogate_Data.vec.payload      = payload
    nexus.Surrogate_Data.vec.opt_names    = opt_names


    generate_vehicle(nexus)
#    print area(poly)
#    print area(poly_translated)



