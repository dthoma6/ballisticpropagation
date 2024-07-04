#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 08:43:26 2024

@author: Dean Thomas
"""

import netCDF4
import numpy as np
import pandas as pd
import os
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt

LIMITS = True      # Change plot xlimits if true
PROPAGATE = True   # Ballistic propagation if true
POINT = 33.        # Point that we ballistically propagate to in Re from earth

############################################################################
#
# Script to ballistically propagate solar wind data
#
# See Mailyan, B., C. Munteanu, and S. Haaland. "What is the best method to 
# calculate the solar wind propagation delay?." Annales geophysicae. Vol. 26. 
# No. 8. Copernicus GmbH, 2008.
#
# This script follows Section 3.1
#
############################################################################

# Download Mother's Day storm solar wind data into the subfolder './input' using:
#
# wget https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_f1m_dscovr_s20240510000000_e20240510235959_p20240511034609_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_f3s_dscovr_s20240510000000_e20240510235959_p20240511034627_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_fc1_dscovr_s20240510000000_e20240510235959_p20240511034549_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_m1m_dscovr_s20240510000000_e20240510235959_p20240511034031_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_m1s_dscovr_s20240510000000_e20240510235959_p20240511034110_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_mg1_dscovr_s20240510000000_e20240510235959_p20240511030128_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_pop_dscovr_s20240510000000_e20240510235959_p20240511034641_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_f1m_dscovr_s20240511000000_e20240511235959_p20240512022214_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_f3s_dscovr_s20240511000000_e20240511235959_p20240512022233_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_fc1_dscovr_s20240511000000_e20240511235959_p20240512022152_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_m1m_dscovr_s20240511000000_e20240511235959_p20240512021629_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_m1s_dscovr_s20240511000000_e20240511235959_p20240512021708_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_mg1_dscovr_s20240511000000_e20240511235959_p20240512013723_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_pop_dscovr_s20240511000000_e20240511235959_p20240512022246_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_f1m_dscovr_s20240512000000_e20240512235959_p20240513022209_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_f3s_dscovr_s20240512000000_e20240512235959_p20240513022231_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_fc1_dscovr_s20240512000000_e20240512235959_p20240513022147_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_m1m_dscovr_s20240512000000_e20240512235959_p20240513021624_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_m1s_dscovr_s20240512000000_e20240512235959_p20240513021703_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_mg1_dscovr_s20240512000000_e20240512235959_p20240513013720_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_pop_dscovr_s20240512000000_e20240512235959_p20240513022245_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_f1m_dscovr_s20240513000000_e20240513235959_p20240514022139_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_f3s_dscovr_s20240513000000_e20240513235959_p20240514022158_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_fc1_dscovr_s20240513000000_e20240513235959_p20240514022117_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_m1m_dscovr_s20240513000000_e20240513235959_p20240514021543_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_m1s_dscovr_s20240513000000_e20240513235959_p20240514021630_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_mg1_dscovr_s20240513000000_e20240513235959_p20240514013653_pub.nc.gz \
#     https://www.ngdc.noaa.gov/dscovr/data/2024/05/oe_pop_dscovr_s20240513000000_e20240513235959_p20240514022213_pub.nc.gz
#
# gunzip the netCDF4 files

# These files, among other variables, contain time, proton_temperature, 
# proton_density, proton_vx_gsm, proton_vy_gsm, proton_vz_gsm
f1mfile = ('./input/oe_f1m_dscovr_s20240510000000_e20240510235959_p20240511034609_pub.nc',
           './input/oe_f1m_dscovr_s20240511000000_e20240511235959_p20240512022214_pub.nc',
           './input/oe_f1m_dscovr_s20240512000000_e20240512235959_p20240513022209_pub.nc',
           './input/oe_f1m_dscovr_s20240513000000_e20240513235959_p20240514022139_pub.nc')

# These files, among other variables, contain time, bx_gsm, by_gsm, bz_gsm
m1mfile = ('./input/oe_m1m_dscovr_s20240510000000_e20240510235959_p20240511034031_pub.nc',
           './input/oe_m1m_dscovr_s20240511000000_e20240511235959_p20240512021629_pub.nc',
           './input/oe_m1m_dscovr_s20240512000000_e20240512235959_p20240513021624_pub.nc',
           './input/oe_m1m_dscovr_s20240513000000_e20240513235959_p20240514021543_pub.nc')
           

# These files, among other variables, contain time, sat_x_gsm (satellite position)
popfile = ('./input/oe_pop_dscovr_s20240510000000_e20240510235959_p20240511034641_pub.nc',
           './input/oe_pop_dscovr_s20240511000000_e20240511235959_p20240512022246_pub.nc',
           './input/oe_pop_dscovr_s20240512000000_e20240512235959_p20240513022245_pub.nc',
           './input/oe_pop_dscovr_s20240513000000_e20240513235959_p20240514022213_pub.nc')

# for name, variable in f1mdata.variables.items():
#     if name == 'time':
#         for attrname in variable.ncattrs():
#             print("{}: {} -- {}".format(name, attrname, variable.getncattr(attrname)))

# We combine the data from multiple files together in these arrays
temp = np.empty( shape=(0) )
rho  = np.empty( shape=(0) )
vx   = np.empty( shape=(0) )
vy   = np.empty( shape=(0) )
vz   = np.empty( shape=(0) )
bx   = np.empty( shape=(0) )
by   = np.empty( shape=(0) )
bz   = np.empty( shape=(0) )
sx   = np.empty( shape=(0) )
time1 = np.empty( shape=(0) )
time2 = np.empty( shape=(0) )
time3 = np.empty( shape=(0) )

# Loop through files and parse data
for i in range(len(f1mfile)):
    f1mdata = netCDF4.Dataset(f1mfile[i])
    f1mtime = np.array( f1mdata.variables['time'] )           # time in unix epoch 
                                                              # aka milliseconds since 1970-01-01T00:00:00Z
    f1mtemp = np.array( f1mdata.variables['proton_temperature'] ) # temperature in K
    f1mrho  = np.array( f1mdata.variables['proton_density'] ) # density in cm^-3
    f1mvx   = np.array( f1mdata.variables['proton_vx_gsm'] )  # velocity in km/sec
    f1mvy   = np.array( f1mdata.variables['proton_vy_gsm'] )
    f1mvz   = np.array( f1mdata.variables['proton_vz_gsm'] )
    
    m1mdata = netCDF4.Dataset(m1mfile[i])
    m1mtime = np.array( m1mdata.variables['time'] )    # time in unix epoch
    m1mbx   = np.array( m1mdata.variables['bx_gsm'] )  # b in nT
    m1mby   = np.array( m1mdata.variables['by_gsm'] )
    m1mbz   = np.array( m1mdata.variables['bz_gsm'] )
    
    popdata = netCDF4.Dataset(popfile[i])
    poptime = np.array( popdata.variables['time'] )      # time in unix epoch
    popsx   = np.array( popdata.variables['sat_x_gsm'] ) # distance in km

    # Add data to arrays created above
    temp = np.concatenate((temp, f1mtemp))
    rho  = np.concatenate((rho , f1mrho ))
    vx   = np.concatenate((vx  , f1mvx  ))
    vy   = np.concatenate((vy  , f1mvy  ))
    vz   = np.concatenate((vz  , f1mvz  ))
    bx   = np.concatenate((bx  , m1mbx  ))
    by   = np.concatenate((by  , m1mby  ))
    bz   = np.concatenate((bz  , m1mbz  ))
    sx   = np.concatenate((sx  , popsx  ))

    time1 = np.concatenate((time1, f1mtime))  # Time for temp, rho, vx, vy, vz
    time2 = np.concatenate((time2, m1mtime))  # Time for bx, by, bz
    time3 = np.concatenate((time3, poptime))  # Time for satellite position

# Verify temp, rho, ... bx, by, bz data have the same timestamps
assert len(time1) == len(time2)
for i in range(len(time1)):
    assert time1[i] == time2[i]

# # Find DSCOVR distance from earth along GSM x axis
# sxmean = np.mean( sx )
# sxstd  = np.std( sx )
# print( 'Mean: ', sxmean, ' StdDev: ', sxstd )

# For ballistic propagation, we need DSCOVR distance from earth along GSM x axis.  
# Satellite position is reported less frequently than solar wind data, so we
# interpolate satellite position
from scipy import interpolate
sx_interp = interpolate.interp1d(time3, sx, fill_value='extrapolate')

# Ballistically propagate the solar wind conditions.  That is, add delay
# for solar wind to travel from DSCOVR satellite position along GSM x axis (km)
# to POINT (Re) along GSM x axis
if PROPAGATE: 
    for i in range(len(time1)):
        distance = POINT*6378.1 - sx_interp(time1[i])
        time1[i] = time1[i] + 1000.*distance/vx[i] # *1000 to get millisecs

# Convert solar wind times to 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'ms'
year1  = np.zeros(len(time1))
month1 = np.zeros(len(time1))
day1   = np.zeros(len(time1))
hour1  = np.zeros(len(time1))
min1   = np.zeros(len(time1))
sec1   = np.zeros(len(time1))
ms1    = np.zeros(len(time1))

for i in range(len(time1)):
    datetime1 = datetime.fromtimestamp(time1[i]/1000, timezone.utc)
    year1[i]  = datetime1.year
    month1[i] = datetime1.month
    day1[i]   = datetime1.day
    hour1[i]  = datetime1.hour
    min1[i]   = datetime1.minute
    sec1[i]   = datetime1.second
    ms1[i]    = datetime1.microsecond
        
df = pd.DataFrame({'Year': year1, 'Month': month1, 'Day': day1, 'Hour': hour1, 
                  'Minute': min1, 'Second': sec1, 'ms': ms1, r'$B_x$ (nT)': bx, 
                  r'$B_y$ (nT)': by, r'$B_z$ (nT)': bz, r'$V_x$ (km/s)': vx, r'$V_y$ (km/s)': vy,
                  r'$V_z$ (km/s)': vz, r'$N$ (${cm}^{-3}$)': rho, r'$T$ (Kelvin)': temp})

# Remove rows with bad data, ie., values = -99999.0
df = df.drop(df[df[r'$T$ (Kelvin)'] < -99998.0].index)

# Add datetime
df['Datetime'] = pd.to_datetime( df[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'ms']] )

if LIMITS:
    xlimits =( datetime(2024,5,10,6,0,0,0), datetime(2024,5,12,0,0,0,0) )
else:
    xlimits =( datetime(2024,5,10,0,0,0,0), datetime(2024,5,14,0,0,0,0) )

# Set some plot configs
plt.rcParams["figure.figsize"] = [12.0,12.0] 
plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.dpi"] = 600
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 22
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})
plt.rcParams['text.latex.preamble'] = r'\usepackage{cmbright}'

fig, ax = plt.subplots(4, 1, sharex=True, sharey=False)

df.plot( x='Datetime', y=[r'$N$ (${cm}^{-3}$)'], \
                xlabel=r'Time (UTC)', \
                ylabel=r'$N$ ({cm}\textsuperscript{-3})', \
                style=['-','r:','--'], \
                xlim=xlimits,\
                grid = False,\
                legend=False,
                ax=ax[0])  
    
df.plot( x='Datetime', y=[r'$T$ (Kelvin)'], \
                xlabel=r'Time (UTC)', \
                ylabel=r'$T$ (Kelvin)', \
                style=['-','r:','--'], \
                xlim=xlimits,\
                grid = False,\
                legend=False,
                ax=ax[1])  
    
df.plot( x='Datetime', y=[r'$V_x$ (km/s)', r'$V_y$ (km/s)', r'$V_z$ (km/s)'], \
                xlabel=r'Time (UTC)', \
                ylabel=r'$V$ (km/s)', \
                style=['-','r:','--', '.'], \
                xlim=xlimits,\
                grid = False,\
                legend=True,
                ax=ax[2])  
ax[2].legend(loc='lower left')

df.plot( x='Datetime', y=[r'$B_x$ (nT)', r'$B_y$ (nT)', r'$B_z$ (nT)'], \
                xlabel=r'Time (UTC)', \
                ylabel=r'$B$ (nT)', \
                style=['-','r:','--', '.'], \
                xlim=xlimits,\
                grid = False,\
                legend=True,
                ax=ax[3])  
ax[3].legend(loc='lower left')

# fig.savefig( os.path.join( info['dir_plots'], "solar_wind_inputs.png" ) )
# # fig.savefig( os.path.join( info['dir_plots'], "solar_wind_inputs.pdf" ) )
# # fig.savefig( os.path.join( info['dir_plots'], "solar_wind_inputs.eps" ) )
# # fig.savefig( os.path.join( info['dir_plots'], "solar_wind_inputs.jpg" ) )
# # fig.savefig( os.path.join( info['dir_plots'], "solar_wind_inputs.tif" ) )
# # fig.savefig( os.path.join( info['dir_plots'], "solar_wind_inputs.svg" ) )
  
    
    
    
