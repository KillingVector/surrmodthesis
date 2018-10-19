# Missions.py
# 
# Created:  Feb 2016, E. Botero
# Modified: 

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import SUAVE
from SUAVE.Core import Units
import time

from Constant_Throttle_Constant_Rate_Anal import Constant_Throttle_Constant_Rate

import numpy as np

# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------
    
def setup(analyses,vehicle):
    
    # the mission container
    missions = SUAVE.Analyses.Mission.Mission.Container()   
    missions.mission = mission(analyses,vehicle)

    return missions  
    
def mission(analyses,vehicle):
    
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission     = SUAVE.Analyses.Mission.Sequential_Segments()
#    mission     = SUAVE.Analyses.Mission.Mission()
    mission.tag = 'mission'

    mission.atmosphere  = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    mission.planet      = SUAVE.Attributes.Planets.Earth()
    
    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments
    
    # base segment
    base_segment = Segments.Segment()     
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery       
    
    # ------------------------------------------------------------------    
    #   Cruise Segment: Constant Dynamic Pressure, Constant Altitude
    # ------------------------------------------------------------------    
    
#    segment = SUAVE.Analyses.Mission.Segments.Cruise.Constant_Dynamic_Pressure_Constant_Altitude(base_segment)
    segment = Constant_Throttle_Constant_Rate()
    segment.tag = "cruise"
    
    # connect vehicle configuration
    segment.analyses.extend(analyses.base)
    
    # segment attributes     
    segment.state.numerics.number_control_points = 10
    segment.dynamic_pressure = 115.0 * Units.pascals
    segment.start_time       = time.strptime("Tue, Jun 21  11:00:00  2017", "%a, %b %d %H:%M:%S %Y",)
    segment.altitude_start       = 30. * Units.km # must be greater than end
    segment.altitude_end         = 29.5 * Units.km # default is zero
    segment.throttle             = 0.5 # can alter if engines, set engines num to zero if glider
    segment.descent_rate         = 2. * Units.m / Units.s # remember z is down
    segment.equivalent_air_speed = 20. * Units.m / Units.s
    segment.charge_ratio     = 1.0
    segment.latitude         = 37.4
    segment.longitude        = -122.15
    segment.state.conditions.frames.wind.body_rotations[:,2] = 125.* Units.degrees 
    
    mission.append_segment(segment)   



    
    return mission
