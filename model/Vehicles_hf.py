# Vehicles.py
# 
# Created:  Feb 2016, E. Botero
# Modified: 

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import SUAVE
import numpy as np
from SUAVE.Core import Units, Data
from SUAVE.Components.Energy.Networks.Solar_Low_Fidelity import Solar_Low_Fidelity
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass
from SUAVE.Methods.Propulsion.electric_motor_sizing import size_from_kv

from surrogate import Surrogate_Data as surrog
from MakeVehicle import generate_vehicle 

from SUAVE.Input_Output.OpenVSP import write
from SUAVE.Input_Output.OpenVSP.get_vsp_areas import get_vsp_areas

# ----------------------------------------------------------------------
#   Define the Vehicle
# ----------------------------------------------------------------------
# make it so i don't need to import EVERYTHING just import surrog and iterate through it (:
def setup(nexus):
    
    base_vehicle = base_setup(nexus)
    configs = configs_setup_hf(base_vehicle)
    
    return configs
    
def base_setup(nexus):

    aerofoil= nexus.surrogate_data.vec.aerofoil
    results = generate_vehicle(nexus)


    s, rootChord, rcp, psl, sqc, do, tw, ttc, aero_geom, A_ref, span_vals, totmass                     = results
    c_bar, ca, cg           = aero_geom
    A_ref_wet, A_ref_pln    = A_ref
    wing_taper, wing_sweep  = span_vals
    
#    print '===== RESULTS'
#    print results
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------ 
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'ModBWB'
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    
    # mass properties
    vehicle.mass_properties.takeoff         = totmass * Units.kg
#    vehicle.mass_properties.operating_empty = totmass * Units.kg
#    vehicle.mass_properties.max_takeoff     = totmass * Units.kg 
    
    # basic parameters
    vehicle.reference_area                    = A_ref_pln * Units.m**2        
    vehicle.envelope.ultimate_load            = 2.0
    vehicle.envelope.limit_load               = 1.5
    vehicle.envelope.maximum_dynamic_pressure = 0.5*1.225*(40.**2.) #Max q

    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------   


    for i in range(0,len(psl)+1):
        airfoil = SUAVE.Components.Wings.Airfoils.Airfoil()
        airfoil.coordinate_file         = './marske1.dat'
        if i == 0:
            wing = SUAVE.Components.Wings.Wing()
            wing.tag = 'main_wing'
    
            wing.aspect_ratio            = s**2 / vehicle.reference_area
            wing.thickness_to_chord      = ttc[0]
            wing.taper                   = wing_taper 
            wing.span_efficiency         = 0.95
            
            wing.spans.projected         = s * Units.m
                
#            print 'ROOT CHORD : ' + str(rootChord)
            wing.chords.root             = rootChord * Units.m
            wing.chords.tip              = wing.taper * wing.chords.root
            wing.chords.mean_aerodynamic = c_bar * Units.m
        
            wing.areas.reference         = A_ref_pln * Units.m**2   
            wing.areas.wetted           = A_ref_wet * Units.m**2
            wing.sweeps.quarter_chord    = wing_sweep * Units.deg

            wing.twists.root             = tw[0] * Units.deg
            wing.twists.tip              = tw[-1] * Units.deg
#            wing.dihedral                = do[0] * Units.deg
        
            wing.origin                  = [0.,0.,0]
            wing.aerodynamic_center      = [ca,0,0]#[ca,0,0]
            wing.center_of_gravity       = [cg,0.,0.]#[cg,0.,0.] 
        
            wing.vertical                = False
            wing.symmetric               = True
            wing.high_lift               = True
        
            wing.dynamic_pressure_ratio  = 1.0

            if len(psl) == 1:
                wing.dihedral               = do[0] * Units.deg
                wing.sweeps.quarter_chord   = sqc[0] * Units.deg

    
            
            # wing segments
        elif i > 0 and len(psl) > 1:
            segment = SUAVE.Components.Wings.Segment()
            segment.tag                   = 'section_' + str(i-1)
            segment.percent_span_location = psl[i-1]
            segment.root_chord_percent    = rcp[i-1]
            segment.dihedral_outboard     = do[i-1] * Units.deg
            segment.sweeps.quarter_chord  = sqc[i-1] * Units.deg
            segment.thickness_to_chord    = ttc[i-1]
            segment.append_airfoil(airfoil)
            wing.Segments.append(segment)  
    
    print wing
    # add to vehicle
    vehicle.append_component(wing)
    
    #------------------------------------------------------------------
    # Propulsor
    #------------------------------------------------------------------

    # build network
    net = Solar_Low_Fidelity()
    net.number_of_engines = 0.#1.
    net.nacelle_diameter  = 0.05
    net.areas             = Data()
    net.areas.wetted      = 0.01*(2*np.pi*0.01/2)
    net.engine_length     = 0.01

    # Component 1 the Sun
    sun = SUAVE.Components.Energy.Processes.Solar_Radiation()
    net.solar_flux = sun

    # Component 2 the solar panels
    panel = SUAVE.Components.Energy.Converters.Solar_Panel()
    panel.ratio                = 0.9
    panel.area                 = vehicle.reference_area * panel.ratio 
    panel.efficiency           = 0.25
    panel.mass_properties.mass = panel.area*(0.60 * Units.kg)
    net.solar_panel            = panel

    # Component 3 the ESC
    esc = SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller()
    esc.efficiency = 0.95 # Gundlach for brushless motors
    net.esc        = esc

    # Component 5 the Propeller
    prop = SUAVE.Components.Energy.Converters.Propeller_Lo_Fid()
    prop.propulsive_efficiency = 0.825
    net.propeller        = prop
    
    # Component 4 the Motor
    motor = SUAVE.Components.Energy.Converters.Motor_Lo_Fid()
    kv                         = 800. * Units['rpm/volt'] # RPM/volt is standard
    motor                      = size_from_kv(motor, kv)    
    motor.gear_ratio           = 1. # Gear ratio, no gearbox
    motor.gearbox_efficiency   = 1. # Gear box efficiency, no gearbox
    motor.motor_efficiency     = 0.825;
    net.motor                  = motor    

    # Component 6 the Payload
    payload = SUAVE.Components.Energy.Peripherals.Payload()
    payload.power_draw           = 0. #Watts 
    payload.mass_properties.mass = 0.0 * Units.kg
    net.payload                  = payload

    # Component 7 the Avionics
    avionics = SUAVE.Components.Energy.Peripherals.Avionics()
    avionics.power_draw = 10. #Watts  
    net.avionics        = avionics      

    # Component 8 the Battery
    bat = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion()
    bat.mass_properties.mass = 5.0  * Units.kg
    bat.specific_energy      = 250. *Units.Wh/Units.kg
    bat.resistance           = 0.003
    bat.iters                = 0
    initialize_from_mass(bat,bat.mass_properties.mass)
    net.battery              = bat

    #Component 9 the system logic controller and MPPT
    logic = SUAVE.Components.Energy.Distributors.Solar_Logic()
    logic.system_voltage  = 18.5
    logic.MPPT_efficiency = 0.95
    net.solar_logic       = logic

    # add the solar network to the vehicle
    vehicle.append_component(net)  

    return vehicle

# ----------------------------------------------------------------------
#   Define the Configurations
# ---------------------------------------------------------------------


def configs_setup_hf(vehicle):
    
    # ------------------------------------------------------------------
    #   Only One Configurations
    # ------------------------------------------------------------------

# for VSP
    configs         = SUAVE.Components.Configs.Config.Container()

    base_config     = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = 'base'

    configs.append(base_config)

    write(vehicle, base_config.tag)

#    print vehicle

    return configs
