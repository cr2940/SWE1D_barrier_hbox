#!/usr/bin/env python
# encoding: utf-8
r"""
Riemann solvers for the shallow water equations.

The available solvers are:
 * Roe - Use Roe averages to caluclate the solution to the Riemann problem
 * HLL - Use a HLL solver
 * Wave redistribution - Use a wave redistribution method to deal with the case
    that the barrier embeded within a cell

.. math::
    q_t + f(q)_x = 0

where

.. math::
    q(x,t) = \left [ \begin{array}{c} h \\ h u \end{array} \right ],

the flux function is

.. math::
    f(q) = \left [ \begin{array}{c} h u \\ hu^2 + 1/2 g h^2 \end{array}\right ].

and :math:`h` is the water column height, :math:`u` the velocity and :math:`g`
is the gravitational acceleration.

:Authors:
    Jiao Li (2018-01-01): Initial version
"""
# ============================================================================
#      Copyright (C) 2020 Chanyang Ryoo <cr2940@columbia.edu>
#
#  Distributed under the terms of the Berkeley Software Distribution (BSD)
#  license
#                     http://www.opensource.org/licenses/
# ============================================================================

import numpy as np

num_eqn = 2
num_waves = 3

def riemanntype(hL, hR, uL, uR, maxiter, drytol, g):
    h_min = min(hR,hL)
    h_max = max(hR,hL)
    delu = uR - uL

    if (h_min <= drytol):
        hm = 0.0
        um = 0.0
        s1m = uR + uL - 2.0 * np.sqrt(g * hR) + 2.0 * np.sqrt(g * hL)
        s2m = uR + uL - 2.0 * np.sqrt(g * hR) + 2.0 * np.sqrt(g * hL)
        if (hL <= 0.0):
            rare2 = True
            rare1 = False
        else:
            rare1 = True
            rare2 = False
    else:
        F_min = delu + 2.0 * (np.sqrt(g * h_min) - np.sqrt(g * h_max))
        F_max = delu + (h_max - h_min) * (np.sqrt(0.5 * g * (h_max + h_min) / (h_max * h_min)))

        if (F_min > 0.0): #2-rarefactions
            hm = (1.0 / (16.0 * g)) * max(0.0, - delu + 2.0 * (np.sqrt(g * hL) + np.sqrt(g * hR)))**2
            um = uL + 2.0 * (np.sqrt(g * hL) - np.sqrt(g * hm))
            s1m = uL + 2.0 * np.sqrt(g * hL) - 3.0 * np.sqrt(g * hm)
            s2m = uR - 2.0 * np.sqrt(g * hR) + 3.0 * np.sqrt(g * hm)
            rare1 = True
            rare2 = True

        elif (F_max <= 0.0): # !2 shocks
            # root finding using a Newton iteration on sqrt(h)===
            h0 = h_max
            for iter in range(maxiter):
                gL = np.sqrt(0.5 * g * (1 / h0 + 1 / hL))
                gR = np.sqrt(0.5 * g * (1 / h0 + 1 / hR))
                F0 = delu + (h0 - hL) * gL + (h0 - hR) * gR
                dfdh = gL - g * (h0 - hL) / (4.0 * (h0**2) * gL) + gR - g * (h0 - hR) / (4.0 * (h0**2) * gR)
                slope = 2.0 * np.sqrt(h0) * dfdh
                h0 = (np.sqrt(h0) - F0 / slope)**2

            hm = h0
            u1m = uL - (hm-hL) * np.sqrt((0.5 * g) * (1 / hm + 1 / hL))
            u2m = uR + (hm - hR) * np.sqrt((0.5 * g) * (1 / hm + 1 / hR))
            um = 0.5 * (u1m + u2m)
            s1m = u1m - np.sqrt(g * hm)
            s2m = u2m + np.sqrt(g * hm)
            rare1 = False
            rare2 = False

        else: #one shock one rarefaction
            h0 = h_min
            for iter in range(maxiter):
                F0 = delu + 2.0 * (np.sqrt(g * h0) - np.sqrt(g * h_max)) + (h0 - h_min) * np.sqrt(0.5 * g * (1 / h0 + 1 / h_min))
                slope = (F_max - F0) / (h_max - h_min)
                h0 = h0 - F0 / slope

            hm = h0
            if (hL > hR):
                um = uL + 2.0 * np.sqrt(g * hL) - 2.0 * np.sqrt(g * hm)
                s1m = uL + 2.0 * np.sqrt(g * hL) - 3.0 * np.sqrt(g * hm)
                s2m = uL + 2.0 * np.sqrt(g * hL) - np.sqrt(g * hm)
                rare1 = True
                rare2 = False
            else:
                s2m = uR - 2.0 * np.sqrt(g * hR) + 3.0 * np.sqrt(g * hm)
                s1m = uR - 2.0 * np.sqrt(g * hR) + np.sqrt(g * hm)
                um = uR - 2.0 * np.sqrt(g * hR) + 2.0 * np.sqrt(g * hm)
                rare2 = True
                rare1 = False

    return hm, um, s1m, s2m, rare1, rare2


def shallow_fwave_1d(q_l, q_r, aux_l, aux_r, problem_data):
    r"""Shallow water Riemann solver using fwaves

    Also includes support for bathymetry but be wary if you think you might have
    dry states as this has not been tested.

    *problem_data* should contain:
     - *grav* - (float) Gravitational constant
     - *sea_level* - (float) Datum from which the dry-state is calculated.

    :Version: 1.0 (2014-09-05)
    """

    g = problem_data['grav']

    num_rp = q_l.shape[1]
    num_eqn = 2
    num_waves = 2

    # Output arrays
    fwave = np.zeros( (num_eqn, num_waves, num_rp) )
    s = np.zeros( (num_waves, num_rp) )
    amdq = np.zeros( (num_eqn, num_rp) )
    apdq = np.zeros( (num_eqn, num_rp) )

    # Extract state
    u_l = np.where(q_l[0,:] - problem_data['sea_level'] > 1e-3,
                   q_l[1,:] / q_l[0,:], 0.0)
    u_r = np.where(q_r[0,:] - problem_data['sea_level'] > 1e-3,
                   q_r[1,:] / q_r[0,:], 0.0)
    phi_l = q_l[0,:] * u_l**2 + 0.5 * g * q_l[0,:]**2
    phi_r = q_r[0,:] * u_r**2 + 0.5 * g * q_r[0,:]**2

    # Speeds
    s[0,:] = u_l - np.sqrt(g * q_l[0,:])
    s[1,:] = u_r + np.sqrt(g * q_r[0,:])

    delta1 = q_r[1,:] - q_l[1,:]
    delta2 = phi_r - phi_l + g * 0.5 * (q_r[0,:] + q_l[0,:]) * (aux_r[0,:] - aux_l[0,:])

    beta1 = (s[1,:] * delta1 - delta2) / (s[1,:] - s[0,:])
    beta2 = (delta2 - s[0,:] * delta1) / (s[1,:] - s[0,:])

    fwave[0,0,:] = beta1
    fwave[1,0,:] = beta1 * s[0,:]
    fwave[0,1,:] = beta2
    fwave[1,1,:] = beta2 * s[1,:]

    for m in range(num_eqn):
        for mw in range(num_waves):
            amdq[m,:] += (s[mw,:] < 0.0) * fwave[m,mw,:]
            apdq[m,:] += (s[mw,:] >= 0.0) * fwave[m,mw,:]

    return fwave, s, amdq, apdq

# robust GEOCLAW augmented riemann solver
def riemann_fwave_1d(hL, hR, huL, huR, bL, bR, uL, uR, phiL, phiR, s1, s2, g):
    num_eqn = 4
    num_waves = 3
    maxiter=5
    drytol = 0.001

    lamb = np.zeros(3) # for third wave case, speeds
    r = np.zeros((3,3))
    sw = np.zeros(3)
    fw = np.zeros((3,3))
    beta = np.zeros(3)
    to1=False
    to2=False

    delh = hR - hL
    delhu = huR - huL
    delb = bR - bL
    delphi = phiR - phiL #+ g * 0.5 * (hL + hR) * delb
    delnorm = delh**2 + delphi**2

    hm, um, s1m, s2m, rare1, rare2 = riemanntype(hL, hR, uL, uR, 1, drytol, g)
    lamb[0] = min(s1, s2m)
    lamb[2] = max(s2, s1m)
    sE1 = lamb[0]
    sE2 = lamb[2]

    hstarHLL = max((huL-huR+sE2*hR-sE1*hL)/(sE2-sE1),0) # middle state in an HLL solve
    rarecorrectortest = True
    rarecorrector = False
    if rarecorrectortest == True:
        sdelta = lamb[2] - lamb[0]
        raremin = 0.5
        raremax = 0.9
        if rare1 == True and sE1*s1m < 0.0:
            raremin = 0.2
        if rare2 == True and sE2*s2m < 0.0:
            raremin = 0.2
        if rare1 == True or rare2 == True:
            rare1st = 3*(np.sqrt(g*hL)-np.sqrt(g*hm))
            rare2nd = 3*(np.sqrt(g*hR)-np.sqrt(g*hm))
            if max(rare1st,rare2nd) > raremin * sdelta and max(rare1st,rare2nd) < raremax*sdelta:
                rarecorrector = True
                if rare1st > rare2nd:
                    lamb[1] = s1m
                    to1 = True
                elif rare2nd > rare1st:
                    lamb[1] = s2m
                    to2 = True
                else:
                    lamb[1] = 0.5 * (s1m+s2m)
        if hstarHLL < min(hL,hR)/5:
            rarecorrector = False
    for mw in range(num_waves):
        r[0,mw] = 1
        r[1,mw] = lamb[mw]
        r[2,mw] = lamb[mw]**2
    if rarecorrector == False:
        lamb[1] = 0.5*(lamb[0]+lamb[2])
        r[0,1] = 0
        r[1,1] = 0
        r[2,1] = 1

 # steady state wave computation:
    criticaltol = max(drytol*g,1e-6)
    criticaltol_2 = np.sqrt(criticaltol)
    deldelh = -delb
    deldelphi = -0.5*(hR + hL) * (g * delb)

    hLstar = hL
    hRstar = hR
    uLstar = uL
    uRstar = uR
    huLstar = uLstar*hLstar
    huRstar = uRstar*hRstar

    conv_tol = 1e-6
    for iter in range(maxiter):
        if min(hLstar,hRstar)<drytol and rarecorrector==True:
            rarecorrector = False
            hLstar = hL
            hRstar = hR
            uLstar = uL
            uRstar = uR
            huLstar = uLstar * hLstar
            huRstar = uRstar * hRstar
            lamb[1] = 0.5*(lamb[0]+lamb[2])
            r[0,1] = 0
            r[1,1] = 0
            r[2,1] = 1

        hbar = max(0.5*(hLstar+hRstar),0)
        s1s2bar = 0.25*(uLstar+uRstar)**2 - g*hbar
        s1s2tilde = max(0,uLstar*uRstar) - g*hbar

        # for sonic computation:
        sonic = False
        if abs(s1s2bar) <= criticaltol:
            sonic = True
        elif s1s2bar*s1s2tilde <= criticaltol**2:
            sonic = True
        elif s1s2bar*sE1*sE2 <= criticaltol**2:
            sonic = True
        elif min(abs(sE1),abs(sE2)) < criticaltol_2:
            sonic = True
        elif sE1 < criticaltol_2 and s1m > -criticaltol_2:
            sonic = True
        elif sE2 > -criticaltol_2 and s2m < criticaltol_2:
            sonic =True
        elif (uL + np.sqrt(g*hL)) * (uR+np.sqrt(g*hR)) < 0:
            sonic = True
        elif (uL - np.sqrt(g*hL)) * (uR-np.sqrt(g*hR)) < 0:
            sonic = True

        if sonic==True:
            deldelh = -delb
        else:
            deldelh = delb*g*hbar/s1s2bar

        # bounds to ensure nonnegativity:
        if sE1 < -criticaltol and sE2 > criticaltol:
            deldelh = min(deldelh,hstarHLL*(sE2-sE1)/sE2)
            deldelh = max(deldelh,hstarHLL*(sE2-sE1)/sE1)
        elif sE1 >= criticaltol:
            deldelh = min(deldelh,hstarHLL*(sE2-sE1)/sE1)
            deldelh = max(deldelh,-hL)
        elif sE2 <= -criticaltol:
            deldelh = min(deldelh,hR)
            deldelh = max(deldelh,hstarHLL*(sE2-sE1)/sE2)

        if sonic == True:
            deldelphi = -g*hbar*delb
        else:
            deldelphi = -delb*g*hbar*s1s2tilde/s1s2bar
        deldelphi=min(deldelphi,g*max(-hLstar*delb,-hRstar*delb))
        deldelphi=max(deldelphi,g*min(-hLstar*delb,-hRstar*delb))

       # solving the linear system
        Del = np.zeros(3)
        Del[0] = delh - deldelh
        Del[1] = delhu
        Del[2] = delphi - deldelphi

        det1 = r[0,0]*(r[1,1]*r[2,2]-r[1,2]*r[2,1])
        det2 = r[0,1] * (r[1,0]*r[2,2] - r[1,2]*r[2,0])
        det3 = r[0,2] * (r[1,0]*r[2,1] - r[1,1]*r[2,0])
        determinant = det1 - det2 + det3

        A = np.zeros((3,3))
        for k in range(3):
            for mw in range(3):
                A[0,mw] = r[0,mw]
                A[1,mw] = r[1,mw]
                A[2,mw] = r[2,mw]
            A[0,k] = Del[0]
            A[1,k] = Del[1]
            A[2,k] = Del[2]
            det1 = A[0,0]*(A[1,1]*A[2,2]-A[1,2]*A[2,1])
            det2 = A[0,1] * (A[1,0]*A[2,2] - A[1,2]*A[2,0])
            det3 = A[0,2] * (A[1,0]*A[2,1] - A[1,1]*A[2,0])
            beta[k] = (det1-det2+det3)/determinant

        if abs(Del[0]**2+Del[2]**2 - delnorm)< conv_tol:
            break
        delnorm = Del[0]**2 + Del[2]**2
        hLstar = hL
        hRstar = hR
        uLstar = uL
        uRstar = uR
        huLstar = uLstar * hLstar
        huRstar = uRstar * hRstar

        for mw in range(3):
            if lamb[mw] < 0:
                hLstar += beta[mw]*r[0,mw]
                huLstar += beta[mw]*r[1,mw]
        for mw in range(2,-1,-1):
            if lamb[mw] > 0:
                hRstar -= beta[mw]*r[0,mw]
                huRstar -= beta[mw]*r[1,mw]

        if hLstar > drytol:
            uLstar = huLstar/hLstar
        else:
            hLstar = max(hLstar,0)
            uLstar = 0.0
        if hRstar > drytol:
            uRstar = huRstar/hRstar
        else:
            hRstar = max(hRstar,0)
            uRstar = 0.0


    for mw in range(3):
        sw[mw] = lamb[mw]
        fw[0,mw] = beta[mw] * r[1,mw]
        fw[1,mw] = beta[mw] * r[2,mw]
        fw[2,mw] = beta[mw] * r[1,mw]

    return fw, sw, to1, to2, beta, Del


def barrier_passing(hL, hR, huL, huR, bL, bR, wall_height, drytol, g, maxiter):

    L2R = False
    R2L = False
    hstarL = 0.0
    hstarR = 0.0
    ustarL = 0.0
    ustarR = 0.0
    if (hL > drytol):
        uL = huL / hL
        hstar,ustar,_,_,_,_ = riemanntype(hL, hL, uL, -uL, maxiter, drytol, g)
        hstartest = max(hL, hstar)
        if (hstartest + bL > 0.5*(bL+bR)+wall_height):
            L2R = True
            hstarL = hstartest + bL - 0.5*(bL+bR) - wall_height
        ustartest = max(uL, ustar)
        ustarL = ustartest

    if (hR > drytol):
        uR = huR / hR
        hstar,ustar,_,_,_,_ = riemanntype(hR, hR, -uR, uR, maxiter, drytol, g)
        hstartest = max(hR, hstar)
        if (hstartest + bR > 0.5*(bL+bR)+wall_height):
            R2L = True
            hstarR = hstartest + bR - 0.5*(bL+bR) - wall_height
        ustartest = max(uR, ustar)
        ustarR = ustartest

    return L2R, R2L, hstarL, hstarR, ustarL, ustarR


def redistribute_fwave(q_l, q_r, aux_l, aux_r, wall_height, drytol, g, maxiter):

    fwave = np.zeros((3, 3, 2))
    s = np.zeros((3, 2))
    betas = np.zeros((3,2))
    amdq = np.zeros((2, 2))
    apdq = np.zeros((2, 2))
    third_wave = np.zeros((3,2))
    DEL = np.zeros(3)

    q_wall = np.zeros((2,3))
    aux_wall = np.zeros((1,3))
    s_wall = np.zeros(2)
    gamma = np.zeros((3,3))
    amdq_wall = np.zeros(2)
    apdq_wall = np.zeros(2)
    lambs = np.zeros(3)

    # hbox method
    q_wall[:,0] = q_l[:].copy()
    q_wall[:,2] = q_r[:].copy()

    aux_wall[0,0] = aux_l.copy()
    aux_wall[0,2] = aux_r.copy()
    aux_wall[0,1] = min(aux_wall[0,0],aux_wall[0,2]) + wall_height

    L2R, R2L, hstarL, hstarR, ustarL, ustarR = barrier_passing(q_wall[0,0], q_wall[0,2], q_wall[1,0], q_wall[1,2], aux_wall[0,0], aux_wall[0,2], wall_height, drytol, g, maxiter)
###########################################
########### ghost state setup #############

    if (L2R==True and R2L==True):
        # wall shorter than bathymetry case:
        if aux_wall[0,1] < aux_wall[0,2]:
            q_wall[0,1] = q_wall[0,2]
            q_wall[1,1] = q_wall[1,2]
            aux_wall[0,1] = aux_wall[0,2]
        elif aux_wall[0,1] <= aux_wall[0,0]:
            q_wall[0,1] = q_wall[0,0]
            q_wall[1,1] = q_wall[1,0]
            aux_wall[0,1] = aux_wall[0,0]

        # wall higher than both bathymetry case (general):
        elif aux_wall[0,1] > aux_wall[0,0] and aux_wall[0,1] > aux_wall[0,2]:
            diff2 = aux_wall[0,1] - aux_wall[0,2]
            diff1 = aux_wall[0,1] - aux_wall[0,0]
            q_wall[0,1] =min( q_wall[0,2] - diff2 , q_wall[0,0] - diff1)
            hts = [q_wall[0,2] - diff2 , q_wall[0,0] - diff1]
            q_wall[1,1] = min(q_wall[1,2],q_wall[1,0])


        else:
            q_wall[0,1] = 0.5*(hstarL+hstarR+aux_wall[0,0]+aux_wall[0,2]) - aux_wall[0,1]# - wall_height #q_wall[0,0]+q_wall[0,2]) - wall_height #
            q_wall[1,1] =q_wall[0,1]*0.5*(ustarL+ustarR)

############################################
############################################
    q_wall_l = q_wall[:,:-1]
    q_wall_r = q_wall[:,1:]
    aux_wall_l = aux_wall[:,:-1]
    aux_wall_r = aux_wall[:,1:]

    for i in range(2):
        hL = q_wall_l[0,i]
        hR = q_wall_r[0,i]
        huL = q_wall_l[1,i]
        huR = q_wall_r[1,i]
        bL = aux_wall_l[0,i]
        bR = aux_wall_r[0,i]
        delb = bR-bL


        # Check wet/dry states
        if (hR > drytol): # right state is not dry
            uR = huR / hR
            phiR = 0.5 * g * hR**2 + huR**2 / hR
        else:
            hR = 0.0
            huR = 0.0
            uR = 0.0
            phiR = 0.0

        if (hL > drytol):
            uL = huL / hL
            phiL = 0.5 * g * hL**2 + huL**2 / hL
        else:
            hL = 0.0
            huL = 0.0
            uL = 0.0
            phiL = 0.0


        if (hL > drytol or hR > drytol):
            wall = np.ones(2)
            if (hR <= drytol):
                hstar,_,_,_,_,_ = riemanntype(hL, hL, uL, -uL, maxiter, drytol, g)
                hstartest = max(hL, hstar)
                if (hstartest + bL <= bR):
                    wall[1] = 0.0
                    hR = hL
                    huR = -huL
                    bR = bL
                    phiR = phiL
                    uR = -uL
                elif (hL + bL <= bR):
                    bR = hL + bL

            if (hL <= drytol):
                hstar,_,_,_,_,_ = riemanntype(hR, hR, -uR, uR, maxiter, drytol, g)
                hstartest = max(hR, hstar)
                if (hstartest + bR <= bL):
                    wall[0] = 0.0
                    hL = hR
                    huL = -huR
                    bL = bR
                    phiL = phiR
                    uL = -uR
                elif (hR + bR <= bL):
                    bL = hR + bR

            sL = uL - np.sqrt(g * hL)
            sR = uR + np.sqrt(g * hR)
            uhat = (np.sqrt(g * hL) * uL + np.sqrt(g * hR) * uR) / (np.sqrt(g * hR) + np.sqrt(g * hL))
            chat = np.sqrt(g * 0.5 * (hR + hL))
            sRoe1 = uhat - chat
            sRoe2 = uhat + chat
            s1 = min(sL, sRoe1)
            s2 = max(sR, sRoe2)

            fw, lamb, to1, to2, beta, Del = riemann_fwave_1d(hL, hR, huL, huR, bL, bR, uL, uR, phiL, phiR, s1, s2, g)
            betas[:,i] = beta
            DEL += Del


            if lamb[1] <0:
                wall1 = wall[0]
            elif lamb[1] >0:
                wall1 = wall[1]
            else:
                wall1 = 0
            s[0,i] = lamb[0] * wall[0]
            s[1,i] = lamb[1] * wall1
            s[2,i] = lamb[2] * wall[1]
            fwave[:,0,i] = fw[:,0] * wall[0]
            fwave[:,1,i] = fw[:,1] * wall1
            fwave[:,2,i] = fw[:,2] * wall[1]
            # print("fw: ", fw)
            for mw in range(3):
                if (s[mw,i] < 0):
                    amdq[:,i] += fwave[:2,mw,i]
                elif (s[mw,i] > 0):
                    apdq[:,i] += fwave[:2,mw,i]
                else:
                    amdq[:,i] += 0.5 * fwave[:2,mw,i]
                    apdq[:,i] += 0.5 * fwave[:2,mw,i]
            print('AMDQ:',amdq[:,i],"APDQ: ",apdq[:,i])
    ################################################################
    lambs[0] = np.min(s) #min(s[0,0],s[0,1]) #0.5*(s[0,0] + s[0,1]) ## min(s[0,0],s[0,1]) #np.min(s) #s[0,0] #np.min(s) #0.5*(s[0,0] + s[0,1])
    lambs[1] =0.5*(np.min(s)+np.max(s))#0.5*(s[1,1]+s[1,0]) #min(s[1,0],s[1,1])#  s[1,0] # 0.5*(s[0,0]+s[2,1]) #0.5*(s[1,1]+s[1,0])
    lambs[2] =np.max(s)#min(s[2,1],s[2,0]) 0.5*(s[2,0] + s[2,1]) # s[2,1] # #min(s[2,1],s[2,0]) #snp.max(s) # [2,1]  0.5*(s[2,0] + s[2,1])
    ##################################################################
    if L2R == False and R2L == False:
        lambs[0]=np.min(s)
        lambs[1] = 0
        lambs[2] = np.max(s)

    s_wall = lambs

    if L2R == True and R2L == True:
        if s_wall[2] - s_wall[0] != 0.0:
            gamma[0,0] = (s_wall[2] *s_wall[1]* np.sum(fwave[0,:,:]) - (s_wall[1]+s_wall[2])*np.sum(fwave[1,:,:]) + np.sum(fwave[2,:,:])) / ((s_wall[2] - s_wall[0])*(s_wall[1]-s_wall[0]))
            gamma[0,2] = (-(s_wall[1]+s_wall[0])*np.sum(fwave[1,:,:]) + s_wall[1]*s_wall[0]*np.sum(fwave[0,:,:]) + np.sum(fwave[2,:,:])) / ((s_wall[2] - s_wall[0])*(s_wall[2]-s_wall[1]))
            gamma[0,1] = (s_wall[0]*s_wall[2]*np.sum(fwave[0,:,:]) - (s_wall[0]+s_wall[2])*np.sum(fwave[1,:,:]) + np.sum(fwave[2,:,:])) / ((s_wall[1]-s_wall[2])*(s_wall[1]-s_wall[0]))
            gamma[1,0] = gamma[0,0] * s_wall[0]
            gamma[2,0] = gamma[1,0] * s_wall[0]
            gamma[1,2] = gamma[0,2] * s_wall[2]
            gamma[2,2] = gamma[1,2] * s_wall[2]
            gamma[1,1] = gamma[0,1] * s_wall[1]
            gamma[2,1] = gamma[1,1] * s_wall[1]
    else:
        if s_wall[2] - s_wall[0] != 0.0:
            gamma[0,0] = (s_wall[2] * np.sum(fwave[0,:,:]) - np.sum(fwave[1,:,:])) / (s_wall[2] - s_wall[0])
            gamma[0,2] = (np.sum(fwave[1,:,:]) - s_wall[0] * np.sum(fwave[0,:,:])) / (s_wall[2] - s_wall[0])
            gamma[2,1] = s_wall [0]*s_wall[2]*np.sum(fwave[0,:,:]) - (s_wall[0]+s_wall[2])*np.sum(fwave[1,:,:]) + np.sum(fwave[2,:,:])
            gamma[1,0] = gamma[0,0] * s_wall[0]
            gamma[2,0] = gamma[1,0] * s_wall[0]
            gamma[1,2] = gamma[0,2] * s_wall[2]
            gamma[2,2] = gamma[1,2] * s_wall[2]

    wave_wall = gamma

    for mw in range(len(s_wall)):
        if s_wall[mw] <0:
            amdq_wall += wave_wall[:2,mw]
        elif s_wall[mw] > 0:
            apdq_wall += wave_wall[:2,mw]
        else:
            amdq_wall += 0.5 * wave_wall[:2,mw]
            apdq_wall += 0.5 * wave_wall[:2,mw]

    return wave_wall[:2,:], s_wall, amdq_wall, apdq_wall

def shallow_fwave_dry_1d(q_l, q_r, aux_l, aux_r, problem_data):
    # print("shallow_fwave_hbox_dry_1d")
    g = problem_data['grav']
    drytol = problem_data['dry_tolerance']
    maxiter = problem_data['max_iteration']

    num_rp = q_l.shape[1]
    num_eqn = 2
    num_waves = 2
    num_ghost = 2


    # Output arrays
    fwave = np.zeros((num_eqn, num_waves, num_rp))
    s = np.zeros((num_waves, num_rp))
    amdq = np.zeros((num_eqn, num_rp))
    apdq = np.zeros((num_eqn, num_rp))

    for i in range(num_rp):
        hL = q_l[0,i]
        hR = q_r[0,i]
        huL = q_l[1,i]
        huR = q_r[1,i]
        bL = aux_l[0,i]
        bR = aux_r[0,i]

        # Check wet/dry states
        if (hR > drytol): # right state is not dry
            uR = huR / hR
            phiR = 0.5 * g * hR**2 + huR**2 / hR
        else:
            hR = 0.0
            huR = 0.0
            uR = 0.0
            phiR = 0.0

        if (hL > drytol):
            uL = huL / hL
            phiL = 0.5 * g * hL**2 + huL**2 / hL
        else:
            hL = 0.0
            huL = 0.0
            uL = 0.0
            phiL = 0.0

        if (hL > drytol or hR > drytol):
            wall = np.ones(2)
            if (hR <= drytol):
                hstar,_,_,_,_ ,_= riemanntype(hL, hL, uL, -uL, maxiter, drytol, g)
                hstartest = max(hL, hstar)
                if (hstartest + bL <= bR):
                    wall[1] = 0.0
                    hR = hL
                    huR = -huL
                    bR = bL
                    phiR = phiL
                    uR = -uL
                elif (hL + bL <= bR):
                    bR = hL + bL

            if (hL <= drytol):
                hstar,_,_,_,_,_ = riemanntype(hR, hR, -uR, uR, maxiter, drytol, g)
                hstartest = max(hR, hstar)
                if (hstartest + bR <= bL):
                    wall[0] = 0.0
                    hL = hR
                    huL = -huR
                    bL = bR
                    phiL = phiR
                    uL = -uR
                elif (hR+ bR <= bL):
                    bL = hR + bR

            sL = uL - np.sqrt(g * hL)
            sR = uR + np.sqrt(g * hR)
            uhat = (np.sqrt(g * hL) * uL + np.sqrt(g * hR) * uR) / (np.sqrt(g * hR) + np.sqrt(g * hL))
            chat = np.sqrt(g * 0.5 * (hR + hL))
            sRoe1 = uhat - chat
            sRoe2 = uhat + chat
            s1 = min(sL, sRoe1)
            s2 = max(sR, sRoe2)
            fw = riemann_fwave_1d(hL, hR, huL, huR, bL, bR, uL, uR, phiL, phiR, s1, s2, g)

            s[0,i] = s1 * wall[0]
            s[1,i] = s2 * wall[1]
            fwave[:,0,i] = fw[:,0] * wall[0]
            fwave[:,1,i] = fw[:,1] * wall[1]

            for mw in range(num_waves):
                if (s[mw,i] < 0):
                    amdq[:,i] += fwave[:,mw,i]
                elif (s[mw,i] > 0):
                    apdq[:,i] += fwave[:,mw,i]
                else:
                    amdq[:,i] += 0.5 * fwave[:,mw,i]
                    apdq[:,i] += 0.5 * fwave[:,mw,i]

    if problem_data['zero_width'] == True:
        nw = problem_data['wall_position']
        wall_height = problem_data['wall_height']
        iw = nw + num_ghost - 1
        fwave[:,:,iw], s[:,iw], amdq[:,iw], apdq[:,iw] = redistribute_fwave(q_l[:,iw:iw+1].copy(), q_r[:,iw:iw+1].copy(), aux_l[0,iw:iw+1].copy(), aux_r[0,iw:iw+1].copy(), wall_height, drytol, g, maxiter)


    return fwave, s, amdq, apdq




def shallow_fwave_hbox_dry_1d(q_l, q_r, aux_l, aux_r, problem_data):
    # print("shallow_fwave_hbox_dry_1d")
    g = problem_data['grav']
    nw = problem_data['wall_position']
    wall_height = problem_data['wall_height']
    drytol = problem_data['dry_tolerance']
    maxiter = problem_data['max_iteration']
    alpha = problem_data['fraction']


    if problem_data['arrival_state'] == False:

        num_rp = q_l.shape[1]
        num_eqn = 2
        num_waves = 3
        num_ghost = 2
        iw = nw + num_ghost - 1

        # Output arrays
        fwave = np.zeros((num_eqn, num_waves, num_rp))
        s = np.zeros((num_waves, num_rp))
        amdq = np.zeros((num_eqn, num_rp))
        apdq = np.zeros((num_eqn, num_rp))

        q_hbox = np.zeros((2,2))
        aux_hbox = np.zeros((1,2))

        ratio1 = 2.0 * alpha / (1 + alpha)
        ratio2 = 2.0 * (1 - alpha) / (2 - alpha)

        q_hbox[:,0] = ratio1 * q_r[:,iw-1] + (1 - ratio1) * q_r[:,iw-2]
        aux_hbox[0,0] = ratio1 * aux_r[0,iw-1] + (1 - ratio1) * aux_r[0,iw-2]
        q_hbox[:,1] = ratio2 * q_l[:,iw+1] + (1 - ratio2) * q_l[:,iw+2]
        aux_hbox[0,1] = ratio2 * aux_l[0,iw+1] + (1 - ratio2) * aux_l[0,iw+2]

        q_l[:,iw] = q_hbox[:,0]
        q_r[:,iw] = q_hbox[:,1]
        aux_l[0,iw] = aux_hbox[0,0]
        aux_r[0,iw] = aux_hbox[0,1]

        for i in range(num_rp):
            hL = q_l[0,i]
            hR = q_r[0,i]
            huL = q_l[1,i]
            huR = q_r[1,i]
            bL = aux_l[0,i]
            bR = aux_r[0,i]

            # Check wet/dry states
            if (hR > drytol): # right state is not dry
                uR = huR / hR
                phiR = 0.5 * g * hR**2 + huR**2 / hR
                # print(hR)
            else:
                hR = 0.0
                huR = 0.0
                uR = 0.0
                phiR = 0.0

            if (hL > drytol):
                uL = huL / hL
                phiL = 0.5 * g * hL**2 + huL**2 / hL
            else:
                hL = 0.0
                huL = 0.0
                uL = 0.0
                phiL = 0.0

            if (hL > drytol or hR > drytol):
                wall = np.ones(2)
                if (hR <= drytol):
                    hstar,_,_,_,_,_ = riemanntype(hL, hL, uL, -uL, maxiter, drytol, g)
                    hstartest = max(hL, hstar)
                    if (hstartest + bL <= bR):
                        wall[1] = 0.0
                        hR = hL
                        huR = -huL
                        bR = bL
                        phiR = phiL
                        uR = -uL
                    elif (hL + bL <= bR):
                        bR = hL + bL

                if (hL <= drytol):
                    hstar,_,_,_,_,_ = riemanntype(hR, hR, -uR, uR, maxiter, drytol, g)
                    hstartest = max(hR, hstar)
                    if (hstartest + bR <= bL):
                        wall[0] = 0.0
                        hL = hR
                        huL = -huR
                        bL = bR
                        phiL = phiR
                        uL = -uR
                    elif (hR + bR <= bL):
                        bL = hR + bR

                sL = uL - np.sqrt(g * hL)
                sR = uR + np.sqrt(g * hR)
                uhat = (np.sqrt(g * hL) * uL + np.sqrt(g * hR) * uR) / (np.sqrt(g * hR) + np.sqrt(g * hL))
                chat = np.sqrt(g * 0.5 * (hR + hL))
                sRoe1 = uhat - chat
                sRoe2 = uhat + chat
                s1 = min(sL, sRoe1)
                s2 = max(sR, sRoe2)
                fw,_,_,_,_,_ = riemann_fwave_1d(hL, hR, huL, huR, bL, bR, uL, uR, phiL, phiR, s1, s2, g)

                s[0,i] = s1 * wall[0]
                s[1,i] = s2 * wall[1]
                fwave[:,0,i] = fw[:2,0] * wall[0]
                fwave[:,1,i] = fw[:2,2] * wall[1]

                for mw in range(num_waves):
                    if (s[mw,i] < 0):
                        amdq[:,i] += fwave[:,mw,i]
                    elif (s[mw,i] > 0):
                        apdq[:,i] += fwave[:,mw,i]
                    else:
                        amdq[:,i] += 0.5 * fwave[:,mw,i]
                        apdq[:,i] += 0.5 * fwave[:,mw,i]

        fwave[:,:,iw], s[:,iw], amdq[:,iw], apdq[:,iw] = redistribute_fwave(q_l[:,iw].copy(), q_r[:,iw].copy(), aux_l[0,iw].copy(), aux_r[0,iw].copy(), wall_height, drytol, g, maxiter)

        return fwave, s, amdq, apdq, q_hbox, aux_hbox

    if problem_data['arrival_state'] == True:
        return redistribute_fwave(q_l, q_r, aux_l, aux_r, wall_height, drytol, g, maxiter)
