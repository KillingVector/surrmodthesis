# Procedure.py
# 
# Created:  Feb 2016, E. Botero
# Modified: 

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import numpy as np

from SUAVE.Core import Units, Data, DataOrdered, ContainerOrdered
from SUAVE.Analyses.Process import Process
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass
from SUAVE.Methods.Propulsion.electric_motor_sizing import size_from_kv
from MakeVehicle import generate_vehicle

#from surrogate import Surrogate_Data as surrog


# ----------------------------------------------------------------------        
#   Setup
# ----------------------------------------------------------------------   

def setup(surrogate):
    # ------------------------------------------------------------------
    #   Analysis Procedure
    # ------------------------------------------------------------------ 
    # size the base config
    procedure = Process()
    procedure.simple_sizing   = simple_sizing
    
    # Size the battery and charge it before the mission
    procedure.weights_battery = weights_battery

    # finalizes the data dependencies
    procedure.finalize        = finalize
    
    # performance studies
    procedure.missions        = Process()
    procedure.missions.base   = simple_mission
    
    # Post process the results
    procedure.post_process    = post_process

    return procedure

# ----------------------------------------------------------------------        
#   Simple Mission
# ----------------------------------------------------------------------    
    
def simple_mission(nexus):
    
    mission = nexus.missions.mission

    # Evaluate the missions and save to results   
    results         = nexus.results
    results.mission = mission.evaluate()
    
    return nexus

# ----------------------------------------------------------------------        
#   Sizing
# ----------------------------------------------------------------------    

def simple_sizing(nexus):

    
    # Pull out the vehicle
    vec     = nexus.vehicle_configurations.base
    num_seg = nexus.surrogate_data.vec.init_vec.num_seg

    # Change the dynamic pressure based on the mission, add a factor of safety   
    vec.envelope.maximum_dynamic_pressure = nexus.missions.mission.segments.cruise.dynamic_pressure*1.2
#    print vec.wings.main_wing
    # pull out variables that are being optimized
    new_vec             = Data()
    new_vec.span        = vec.wings.main_wing.spans.projected
    new_vec.root_chord  = vec.wings.main_wing.chords.root
    new_vec.sqc         = []
    new_vec.rcp         = []
    new_vec.psl         = []
    new_vec.do          = []
    new_vec.tw          = []
    new_vec.ttc         = []
  

    print '================'
    in_segs = False
    for item in vec.wings.main_wing:
        if type(item) == type(vec.wings.main_wing.Segments):    # check if item is wing segment
            try:
                for i in range(0,num_seg):
                    key     = 'section_' + str(i)
                    section = getattr(item,key)
    #                print 'section_'+str(i) + '\n'+str(section)
                    new_vec.sqc.append(np.rad2deg(section.sweeps.quarter_chord))
                    new_vec.rcp.append(section.root_chord_percent)
                    new_vec.psl.append(section.percent_span_location)
                    new_vec.do.append(np.rad2deg(section.dihedral_outboard))
                    new_vec.ttc.append(section.thickness_to_chord)
                    in_segs = True
            except Exception as err:
                print err
                print 'Not a wing segment, implied whole wing'
    if new_vec.psl == []:
        new_vec.sqc.append(np.rad2deg(vec.wings.main_wing.sweeps.quarter_chord))
        new_vec.rcp.append(1.)
        new_vec.rcp.append(vec.wings.main_wing.taper)
        new_vec.psl.append(0.)
        new_vec.do.append(np.rad2deg(vec.wings.main_wing.dihedral))
        new_vec.ttc.append(vec.wings.main_wing.thickness_to_chord)

                

    new_vec.tw.append(np.rad2deg(vec.wings.main_wing.twists.root))
    new_vec.tw.append(np.rad2deg(vec.wings.main_wing.twists.tip))

    if in_segs:
        new_vec.rcp.append(vec.wings.main_wing.taper)
        
    #       logic checks
    # span 2* chord at min
    if new_vec.span < 2 * new_vec.root_chord:
        new_vec.span = 2 * new_vec.root_chord
    # no inboard psl less than psl outboard
    if len(new_vec.psl) > 1:
        for i in range(1,len(new_vec.psl)):
            if new_vec.psl[i] < new_vec.psl[i-1]:
                new_vec.psl[i] = new_vec.psl[i-1] + (1 - new_vec.psl[i-1])/2    
                # should find intermediate location
    # no outboard chord section larger than inboard (wing tear)
    for i in range(1,len(new_vec.rcp)):
        if new_vec.rcp[i] > new_vec.rcp[i-1]:
            new_vec.rcp[i]  = new_vec.rcp[i-1]



    #   assign new iteration of vehicle to surrog data vehicle store
    nexus.surrogate_data.vec.make_vec = new_vec

    results = None


    results = generate_vehicle(nexus)

    s, rootChord, rcp, psl, sqc, do, tw, ttc, aero_geom, A_ref, span_vals, totmass                     = results
    c_bar, ca, cg           = aero_geom
    A_ref_wet, A_ref_pln    = A_ref
    wing_taper, wing_sweep  = span_vals



    """
        UPDATE ALL NECESSARY THINGS (:

    """

    # update all wing structure values
    vec.mass_properties.takeoff         = totmass * Units.kg
#    vec.mass_properties.operating_empty = totmass * Units.kg
#    vec.mass_properties.max_takeoff     = totmass * Units.kg

    vec.reference_area          = A_ref_pln * Units.m**2

    vw                          = vec.wings.main_wing
    vw.spans.projected          = s * Units.m

    vw.taper                    = wing_taper
    vw.chords.root              = rootChord * Units.m
    vw.chords.tip               = wing_taper*vw.chords.root
    vw.chords.mean_aerodynamic   = c_bar * Units.m

    vw.areas.reference          = A_ref_pln * Units.m**2
    vw.areas.wetted             = A_ref_wet * Units.m**2
    vw.aspect_ratio            = s**2 / (A_ref_pln)

    vw.sweeps.quarter_chord      = wing_sweep * Units.deg

    vw.twists.root              = new_vec.tw[0] * Units.deg
    vw.twists.tip               = new_vec.tw[-1] *Units.deg

    if len(psl) == 1:
        vw.dihedral               = do[0] * Units.deg
        vw.sweeps.quarter_chord   = sqc[0] * Units.deg



    # update all segment values
    if len(psl) > 1:
        for item in vec.wings.main_wing:
            if type(item) == type(vec.wings.main_wing.Segments):    # check if item is wing segment
                for i in range(0,num_seg):
                    key     = 'section_' + str(i)
    #                print item
                    segment = getattr(item,key)
    #                print segment.tag
                    segment.percent_span_location = psl[i]
                    segment.root_chord_percent    = rcp[i]
                    segment.dihedral_outboard     = do[i] * Units.deg
                    segment.sweeps.quarter_chord  = sqc[i] * Units.deg
                    segment.thickness_to_chord    = ttc[i]
#                print 'section'+str(i) + '\n'+str(segment)

    # Size solar panel area
    wing_area                   = vec.reference_area
    spanel                      = vec.propulsors.network.solar_panel
    sratio                      = spanel.ratio
    solar_area                  = wing_area*sratio
    spanel.area                 = solar_area
    spanel.mass_properties.mass = solar_area*(0.60 * Units.kg)    
    
    # Resize the motor
    motor = vec.propulsors.network.motor
    kv    = motor.speed_constant
    motor = size_from_kv(motor, kv)    
    
    # diff the new data
    vec.store_diff()
    return nexus

# ----------------------------------------------------------------------
#   Calculate weights and charge the battery
# ---------------------------------------------------------------------- 

def weights_battery(nexus):

    # Evaluate weights for all of the configurations
    config = nexus.analyses.base
    config.weights.evaluate() 
    
    vec     = nexus.vehicle_configurations.base
    payload = vec.propulsors.network.payload.mass_properties.mass  
    msolar  = vec.propulsors.network.solar_panel.mass_properties.mass
    MTOW    = vec.mass_properties.max_takeoff
    empty   = vec.weight_breakdown.empty
    mmotor  = vec.propulsors.network.motor.mass_properties.mass
    
    # Calculate battery mass
    batmass = MTOW - empty - payload - msolar -mmotor
    bat     = vec.propulsors.network.battery
    initialize_from_mass(bat,batmass)
    vec.propulsors.network.battery.mass_properties.mass = batmass
        
    # Set Battery Charge
    maxcharge = nexus.vehicle_configurations.base.propulsors.network.battery.max_energy
    charge    = maxcharge
    
    nexus.missions.mission.segments.cruise.battery_energy = charge 

    return nexus
    
# ----------------------------------------------------------------------
#   Finalizing Function
# ----------------------------------------------------------------------    

def finalize(nexus):
    
    nexus.analyses.finalize()   
    
    return nexus         

# ----------------------------------------------------------------------
#   Post Process results to give back to the optimizer
# ----------------------------------------------------------------------   

#   from stack exchange, this lets me extract order of magnitude to apply
#   proper penalties
def magnitude(x):
    return int(math.floor(math.log10(x)))


def post_process(nexus):
    inps = nexus.optimization_problem.inputs
#    print 'Inputs : \n' + str(inps)
    
    # Unpack
    mis = nexus.missions.mission.segments.cruise
    vec = nexus.vehicle_configurations.base
    res = nexus.results.mission.segments.cruise.conditions

    
    # Final Energy
    maxcharge    = vec.propulsors.network.battery.max_energy
    
    # Energy constraints, the battery doesn't go to zero anywhere, using a P norm
    p                    = 8.    
    energies             = res.propulsion.battery_energy[:,0]/np.abs(maxcharge)
    energies[energies>0] = 0.0 # Exclude the values greater than zero
    energy_constraint    = np.sum((np.abs(energies)**p))**(1/p) 

    # CL max constraint, it is the same throughout the mission
    CL = res.aerodynamics.lift_coefficient[0]
    print 'Coefficient of Lift : ' + str(CL)
    # lift, drag, velocity, mass
    # check AoA
    aoa_rad = np.average(res.aerodynamics.angle_of_attack)
    aoa_deg = np.rad2deg(res.aerodynamics.angle_of_attack[0])
    aoa_std = np.rad2deg(np.std(res.aerodynamics.angle_of_attack))
    print 'Angle of attack : ' + str(aoa_deg) + ' --st dev : ' \
    + str(aoa_std)
    print 'Velocity vector average : ' + str(np.average(res.frames.inertial.velocity_vector[:,0]))
#    print res.frames.wind.drag_force_vector
#    print res.frames.wind.lift_force_vector
    drag    = -res.frames.wind.drag_force_vector[0,0] # added
    lift    = -res.frames.wind.lift_force_vector[0,2]
    lond    = lift / drag
    donl    = drag / lift
    velVec  = res.frames.inertial.velocity_vector[0,0]
    invVel  = 1 / velVec
    mass    = vec.mass_properties.takeoff
    print 'post processing L/D : ' + str(lond) + '\n====\n\n'
    
    # calculate objective + constraint
    CL_max                  = .8
    mass_max                = 200.
    aoa_std_max             = 0.25
    pen_scale_aoa           = 1.
    pen_val_aoa             = (max(0,aoa_std-aoa_std_max))**2
    pen_scale_cl            = 1.0e-1
    pen_val_cl              = (max(0,CL-CL_max))**2
    pen_scale_m             = 2.5e-3
    pen_val_m               = (max(0,mass-mass_max))**2
    penalty_cl              = pen_scale_cl * pen_val_cl
    penalty_aoa             = pen_scale_aoa*pen_val_aoa
    print 'CL penalty : ' + str(penalty_cl)
    penalty_m               = pen_scale_m * pen_val_m
    obj_ld                  = -lond + penalty_cl +penalty_aoa
    obj_mass                = mass + penalty_m



    summary = nexus.summary
    summary.CL                = 1.2 - CL

    if not hasattr(summary,'aoa'):
        summary.aoa = []

    summary.aoa.append([aoa_deg, aoa_std])
    summary.penalty           = penalty_cl
    print 'AOA : ' + str(summary.aoa)
    summary.invVel            = invVel
    summary.obj_ld            = obj_ld
    summary.obj_mass          = obj_mass
    summary.mass              = mass

    summary.energy_constraint = energy_constraint
    summary.throttle_min      = res.propulsion.throttle[0]
    summary.throttle_max      = 0.9 - res.propulsion.throttle[0]
    summary.nothing           = 0.0
    
    return nexus    
