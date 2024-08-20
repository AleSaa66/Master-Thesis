#!/usr/bin/env python
# coding: utf-8
"""
Program that solves the stellar equilibrium equations for neutron stars using EoS at zero temperature: 
APR4, SLy, AP4, MPA1, MS2, BSk19, BSk20 y BSk21, in General Relativity for different values of central density.

The values of the dimensionless radius and mass found in each numerical integration process are stored, together
with the profiles of the pressure, energy dnesity, adiabatic index and metric coefficients as a function of the 
radial dimensionless x coordinate.

by: Alejandro Saavedra
https://github.com/AleSaa66
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sp
import Realistic_EoS as re
import h5py
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from tqdm import tqdm  #Progress bar

#Physical constants:
c_light = sp.speed_of_light           #[m]
G = sp.gravitational_constant         #[N m^2 kg^-1]
m_n = sp.neutron_mass                 #[kg]
k = sp.Boltzmann                      #[J s]
h = sp.hbar                           #[J s]

#Constant to express the dimensionless pressure and energy density:
K_mks = (m_n**4*c_light**5)/(32*np.pi**2*h**3) #[N m^-2]
K_cgs = 10*K_mks                               #[dyn cm^-2] 

#Definition of the ODEs system:
def root(xi,a,zeta_p,eos):
    return eos(xi,a) - zeta_p

def epsilon(P,a,eos):
    #Expresar P en la variable zeta:
    zeta_p = np.log10(K_cgs*P)
    
    #Encontrar la densidad:
    xi_p = fsolve(root,10,args=(a,zeta_p,eos))[0]
    ep = (c_light**2/K_mks)*(10**(xi_p + 3))

    return ep

def dfdx(x,f,a,eos):
    #f = [m,alpha,P]:
    m = f[0]
    alpha = f[1] 
    P = f[2]
    ep = epsilon(P,a,eos)

    #ODEs:
    dmdx = x**2*ep 
    dadx = 2*(m + x**3*P)/(x*(x-2*m))
    dPdx = - (ep + P)*(m + x**3*P)/(x*(x-2*m))

    return [dmdx,dadx,dPdx]

#Adiabatic Index:
def Gamma(ep,P,a,derivative_eos):
    #Mass density:
    rho = ep*K_mks/c_light**2
    xi = np.log10(0.001*rho)     #It was multiplied by 10^-3 so the density is in CGS units.

    #Calculus of the adiabatic index:
    gamma = derivative_eos(xi,a)*(1 + P/ep)

    return gamma    

#Event to stop the integration at P = 0:
def stop(x,f,a,eos): 
    #f = [m,alpha,P]:
    m = f[0]
    alpha = f[1]
    P = f[2]

    return P

stop.terminal = True

#Function to save the data in HDF5:
def save_to_h5(file_name, array_name, array):
    with h5py.File(file_name, 'a') as f:
        x5 = f.create_dataset(array_name,shape = (len(array),), dtype = "float64")
        x5[:] = array


#Parameters of the ODEs:
a_EoS = [re.a2,re.a3,re.a4,re.a5,re.a6,re.a7,re.a8,re.a9]
EoS = [re.AMM,re.AMM,re.AMM,re.FSA,re.FSA,re.BSK,re.BSK,re.BSK]
derivative_EoS = [re.dzdxi_AMM,re.dzdxi_AMM,re.dzdxi_AMM,re.dzdxi_FSA,re.dzdxi_FSA,re.dzdxi_BSK,re.dzdxi_BSK,re.dzdxi_BSK]
txt_EoS = ['AP4','MPA1','MS2','SLy','APR4','BSK19','BSK20','BSK21']

#Values of rho_0 (central mass density):
rho0_cgs = np.geomspace(2e14,4.8e17,5000)           #(g/cm^3)
rho0_mks = rho0_cgs*1e3                             #(kg/m^3) 

#Log of mass density                            
xi_0 = np.log10(rho0_cgs)

#Domain x for integration:
x = np.linspace(1.0e-30,4.0,10**5)

for i, func in tqdm(enumerate(EoS)):
    print('Starting the integration for '+txt_EoS[i])

    #Lists to save the dimensionaless radius and mass:
    tx_1 = []       
    tM = []       

    #Log of pressure:
    zeta_0 = func(xi_0,a_EoS[i])

    #Values of the dimensionless pressure and energy density at the origin:
    P_0 = 10**(zeta_0)/K_cgs
    ep_0 = rho0_mks*c_light**2/K_mks

    #Vectorized density function:
    ep_vec = np.vectorize(lambda P: epsilon(P,a_EoS[i],EoS[i]))

    #Numerical integration for each P_0:
    for j in tqdm(range(len(P_0))):
        #Interior solution:
        tx = np.zeros(len(x))
        talpha = np.zeros(len(x))
        tbeta = np.zeros(len(x))
        tep = np.zeros(len(x))
        tP = np.zeros(len(x))
        tgamma = np.zeros(len(x))
        
        #Boundary conditions:
        f0 = [0.0,0.0,P_0[j]]

        #Inetrgation:
        sol = solve_ivp(dfdx, t_span = [x[0],x[-1]], y0 = f0, args = (a_EoS[i],func), method= 'RK45', events = stop,t_eval = x, rtol = 1e-10, atol = 1e-15)

        #Zero found by solve_ivp:
        x_1 = sol.t[-1]
     
        #Calculus of m(x_1):
        M = sol.y[0][-1]

        #Data storage:
        tx_1.append(x_1)
        tM.append(M)

        #Continuity of the interior solution with the outer solution: (alpha(x) = alpha_num(x) + cte)
        cte = np.log(1 - 2*M/x_1) - sol.y[1][-1]

        #Solution found:
        index = len(sol.t)

        tx[:index] = sol.t
        talpha[:index] = sol.y[1] + cte*np.ones(index)
        tbeta[:index] = -1*np.log(1 - 2*sol.y[0]/sol.t)
        tP[:index] = sol.y[2]
        tep[:index] = ep_vec(sol.y[2]) 
        tgamma[:index] = Gamma(tep[:index],tP[:index],a_EoS[i],derivative_EoS[i])
       
        #Save the values of the field for each central density:
        if j == 0:
            with h5py.File('x-sol-GR-'+txt_EoS[i]+'.h5', 'w') as f:
                x5 = f.create_dataset(f'x_{i}', data=tx[:index])

            with h5py.File('alpha-sol-GR-'+txt_EoS[i]+'.h5', 'w') as f:
                alpha5 = f.create_dataset(f'alpha_{i}', data=talpha[:index])
            
            with h5py.File('beta-sol-GR-'+txt_EoS[i]+'.h5', 'w') as f:
                beta5 = f.create_dataset(f'beta_{i}', data=tbeta[:index])

            with h5py.File('P-sol-GR-'+txt_EoS[i]+'.h5', 'w') as f:
                P5 = f.create_dataset(f'P_{i}', data=tP[:index])

            with h5py.File('ep-sol-GR-'+txt_EoS[i]+'.h5', 'w') as f:
                ep5 = f.create_dataset(f'ep_{i}', data=tep[:index])
            
            with h5py.File('gamma-sol-GR-'+txt_EoS[i]+'.h5', 'w') as f:
                gamma5 = f.create_dataset(f'gamma_{i}', data=tgamma[:index])
                
            del tx, talpha, tbeta, tep, tP, tgamma
                
        else:
            save_to_h5('x-sol-GR-'+txt_EoS[i]+'.h5',f'x_{j}',tx[:index])
            save_to_h5('alpha-sol-GR-'+txt_EoS[i]+'.h5',f'alpha_{j}',talpha[:index])
            save_to_h5('beta-sol-GR-'+txt_EoS[i]+'.h5',f'beta_{j}',tbeta[:index])
            save_to_h5('P-sol-GR-'+txt_EoS[i]+'.h5',f'P_{j}',tP[:index])
            save_to_h5('ep-sol-GR-'+txt_EoS[i]+'.h5',f'ep_{j}',tep[:index])
            save_to_h5('gamma-sol-GR-'+txt_EoS[i]+'.h5',f'gamma_{j}',tgamma[:index])

            del tx, talpha, tbeta, tep, tP,tgamma

    #Save the data of the dimensionless radius and mass of the star:
    atx_1 = np.array(tx_1)
    atM = np.array(tM)

    np.savez_compressed('Radii-EoS-'+txt_EoS[i]+'-GR', y = atx_1)
    np.savez_compressed('Mass-EoS-'+txt_EoS[i]+'-GR', y = atM)
    np.savez_compressed('Central_density-EoS-'+txt_EoS[i]+'-GR', y = rho0_cgs)

    print('Ending the integration for '+txt_EoS[i])

