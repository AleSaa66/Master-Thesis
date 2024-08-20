"""
Module with the analytical expressions and parameters of the realistic equation of states for  
neutron stars at zero temperature of the form zeta = zeta(xi), where zeta = log(P/dyn cm^-2) and xi = log(rho /g cm^-3).

by: Alejandro Saavedra
https://github.com/AleSaa66
"""
import numpy as np

#EoS definitions:
def AMM(xi,a):                 #AP4, MPA1, MS2
    #Paper: Can Gungor and K. Yavuz Eksi. "Analytical Representation for Equations of State of Dense Matter". arXiv:1108.2166.
    
    #Auxiliary function:
    f_0 = lambda x: np.exp(-x)/(np.exp(-x) + 1)

    #Extra parameters:
    c = np.array([10.6557,3.7863,0.8124,0.6823,3.5279,11.8100,12.0584,1.4663,3.4952,11.8007,14.4114,14.4081])

    #Regimes:
    zeta_low = (c[0] + c[1]*(xi - c[2])**c[3])*f_0(c[4]*(xi - c[5])) + (c[6] + c[7]*xi)*f_0(c[8]*(c[9] - xi))
    zeta_high = (a[2] + a[3]*xi)*f_0(a[4]*(a[5] - xi)) + (a[6] + a[7]*xi + a[8]*xi**2)*f_0(a[9]*(a[10] - xi))

    #zeta(xi):
    zeta = zeta_low*f_0(a[0]*(xi - c[10])) + f_0(a[1]*(c[11] - xi))*zeta_high

    return zeta

def FSA(xi,a):                   #FSP, SLy, APR4
    #Paper: P. Haensel and A. Y. Potekhin. "Analytical representations of unified equations of state of neutron-star matter". doi: 10.1051/0004-6361:20041722.
    
    #Auxiliary function:
    f_0 = lambda x: np.exp(-x)/(np.exp(-x) + 1)

    #zeta(xi):
    zeta = ((a[0] + a[1]*xi + a[2]*xi**3)*f_0(a[4]*(xi - a[5]))/(1 + a[3]*xi)) + (a[6] + a[7]*xi)*f_0(a[8]*(a[9] - xi)) + (a[10] + a[11]*xi)*f_0(a[12]*(a[13] - xi)) + (a[14] + a[15]*xi)*f_0(a[16]*(a[17] - xi))

    return zeta

def BSK(xi,a):                   #BSk19, BSk20, BSk21   
    #Paper: A. Y. Potekhin et al. "Analytical representations of unified equations of state for neutron-star matter". doi: 10.1051/0004-6361/201321697.
                      
    #Auxiliary function:
    f_0 = lambda x: np.exp(-x)/(np.exp(-x) + 1)

    #zeta(xi):
    zeta = ((a[0] + a[1]*xi + a[2]*xi**3)/(1 + a[3]*xi))*f_0(a[4]*(xi - a[5])) + (a[6] + a[7]*xi)*f_0(a[8]*(a[5] - xi)) + (a[9] + a[10]*xi)*f_0(a[11]*(a[12] - xi)) + (a[13] + a[14]*xi)*f_0(a[15]*(a[16] - xi)) + a[17]/(1 + (a[18]*(xi - a[19]))**2) + a[20]/(1 + (a[21]*(xi - a[22]))**2)

    return zeta

#Derivatives of EoS:
def dzdxi_AMM(xi,a):                 #AP4, MPA1, MS2
    #Auxiliary function:
    f_0 = lambda x: np.exp(-x)/(np.exp(-x) + 1)

    #Extra parameters:
    c = np.array([10.6557,3.7863,0.8124,0.6823,3.5279,11.8100,12.0584,1.4663,3.4952,11.8007,14.4114,14.4081])

    #dzeta/dxi:
    dzdxi_1 = (c[1]*(xi - c[2])**c[3]*c[3]*f_0(c[4]*(xi - c[5]))/(xi - c[2]) - (c[0] + c[1]*(xi - c[2])**c[3]*c[4]*np.exp(c[4]*(xi - c[5])))*f_0(c[4]*(xi - c[5]))**2 + c[7]*f_0(c[8]*(c[9] - xi)) + (c[7]*xi + c[6])*c[8]*np.exp(c[8]*(c[9] - xi))*f_0(c[8]*(c[9] - xi))**2)*f_0(a[0]*(xi - c[10]))
    dzdxi_2 = - ((c[0] + c[1]*(xi - c[2])**c[3])*f_0(c[4]*(xi - c[5])) + (c[7]*xi + c[6])*f_0(c[8]*(c[9] - xi)))*a[0]*np.exp(a[0]*(xi - c[10]))*f_0(a[0]*(xi - c[10]))**2
    dzdxi_3 = ((a[3]*xi + a[2])*f_0(a[4]*(a[5] - xi)) + (a[8]*xi**2 + a[7]*xi + a[6])*f_0(a[9]*(a[10] -xi)))*a[1]*np.exp(a[1]*(c[11] - xi))*f_0(a[1]*(c[11] - xi))**2
    dzdxi_4 = (a[3]*f_0(a[4]*(a[5] - xi)) + (a[3]*xi + a[2])*a[4]*np.exp(a[4]*(a[5] - xi))*f_0(a[4]*(a[5] - xi))**2 + (2*a[8]*xi + a[7])*f_0(a[9]*(a[10] - xi)) + (a[8]*xi**2 + a[7]*xi + a[6])*a[9]*np.exp(a[9]*(a[10] - xi))*f_0(a[9]*(a[10] - xi))**2)*f_0(a[1]*(c[11] - xi))

    dzdxi = dzdxi_1 + dzdxi_2 + dzdxi_3 + dzdxi_4

    return dzdxi

def dzdxi_FSA(xi,a):                   #FSP, SLy, APR4
    #Auxiliary function:
    f_0 = lambda x: np.exp(-x)/(np.exp(-x) + 1)

    #dzeta/dxi:
    dzdxi = ((3*a[2]*xi**2 + a[1])/(a[3]*xi + 1))*f_0(a[4]*(xi - a[5])) - (a[3]*(a[2]*xi**3 + a[1]*xi + a[0])/(a[3]*xi + 1)**2)*f_0(a[4]*(xi - a[5])) - ((a[2]*xi**3 + a[1]*xi + a[0])*a[4]*np.exp(a[4]*(xi - a[5]))/(a[3]*xi + 1))*f_0(a[4]*(xi - a[5]))**2 + a[7]*f_0(a[8]*(a[9] - xi)) + (a[7]*xi + a[6])*a[8]*np.exp(a[8]*(a[9] - xi))*f_0(a[8]*(a[9] - xi))**2 + a[11]*f_0(a[12]*(a[13] - xi)) + (a[11]*xi + a[10])*a[12]*np.exp(a[12]*(a[13] - xi))*f_0(a[12]*(a[13] - xi))**2 + a[15]*f_0(a[16]*(a[17] - xi)) + (a[15]*xi + a[14])*a[16]*np.exp(a[16]*(a[17] - xi))*f_0(a[16]*(a[17] - xi))**2

    return dzdxi

def dzdxi_BSK(xi,a):                   #BSk19, BSk20, BSk21                     
    #Auxiliary function:
    f_0 = lambda x: np.exp(-x)/(np.exp(-x) + 1)

    #dzeta/dxi:
    dzdxi = (3*a[2]*xi**2 + a[1])*f_0(a[4]*(xi - a[5]))/(a[3]*xi + 1) - (a[2]*xi**3 + a[1]*xi + a[0])*a[3]*f_0(a[4]*(xi - a[5]))/(a[3]*xi + 1)**2 - (a[2]*xi**3 + a[1]*xi + a[0])*a[4]*np.exp(a[4]*(xi - a[5]))*f_0(a[4]*(xi - a[5]))**2/(a[3]*xi + 1) + a[7]*f_0(a[8]*(a[5] - xi)) + (a[7]*xi + a[6])*a[8]*np.exp(a[8]*(a[5] - xi))*f_0(a[8]*(a[5] - xi))**2 + a[10]*f_0(a[11]*(a[12] - xi)) + (a[10]*xi + a[9])*a[11]*np.exp(a[11]*(a[12] - xi))*f_0(a[11]*(a[12] - xi))**2 + a[14]*f_0(a[15]*(a[16] - xi)) + (a[14]*xi + a[13])*a[15]*np.exp(a[15]*(a[16] - xi))*f_0(a[15]*(a[16] - xi))**2 - 2*a[17]*a[18]**2*(xi - a[19])/(1 + a[18]**2*(xi - a[19])**2)**2 - 2*a[20]*a[21]**2*(xi - a[22])/(1 + a[21]**2*(xi - a[22])**2)**2

    return dzdxi

#Parameters of the EoS:

#AP1:
a1 = np.array([4.3290,4.3622,138.1760,-10.1093,6.0097,14.0120,-411.1380,48.0721,-1.1630,4.7514,10.73234])

#AP4:
a2 = np.array([4.3290,4.3622,9.1131,-0.4751,3.4614,14.8800,21.3141,0.1023,0.0495,4.9401,10.2957])

#mpa1:
a3 = np.array([5.2934,5.3319,87.7901,-5.8466,2.7232,15.0804,428.4130,-57.6403,2.0957,5.0588,10.2727])

#ms2:
a4 = np.array([14.0084,13.8422,16.5970,-1.0943,5.6701,14.8169,-56.3794,9.6159,-0.2332,-3.8369,23.1860])

#SLy:
a5 = np.array([6.22,6.121,0.005925,0.16326,6.48,11.4971,19.105,0.8938,6.54,11.4950,-22.775,1.5707,4.3,14.08,27.80,-1.653,1.50,14.67])

#APR:
a6 = np.array([6.22,6.121,0.006035,0.16354,4.73,11.5831,12.589,1.4365,4.75,11.5756,-42.489,3.8175,2.3,14.81,29.80,-2.976,1.99,14.93])

#BSk19:
a7 = np.array([3.916,7.701,0.00858,0.22114,3.269,11.964,13.349,1.3683,3.254,-12.953,0.9237,6.20,14.383,16.693,-1.0514,2.486,15.362,0.085,6.23,11.68,-0.029,20.1,14.19])

#BSk20:
a8 = np.array([4.078,7.587,0.00839,0.21695,3.614,11.942,13.751,1.3373,3.606,-22.996,1.6229,4.88,14.274,23.560,-1.5564,2.095,15.294,0.084,6.36,11.67,-0.042,14.8,14.18])

#BSk21:
a9 = np.array([4.857,6.981,0.00706,0.19351,4.085,12.065,10.521,1.5905,4.104,-28.726,2.0845,4.89,14.302,22.881,-1.7690,0.989,15.313,0.091,4.68,11.65,-0.086,10.0,14.15])