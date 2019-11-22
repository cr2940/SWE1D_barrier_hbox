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
#      Copyright (C) 2018 Jiao Li <jl4170@columbia.edu>
#
#  Distributed under the terms of the Berkeley Software Distribution (BSD)
#  license
#                     http://www.opensource.org/licenses/
# ============================================================================

import numpy as np

num_eqn = 2
num_waves = 2

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

    return hm, s1m, s2m, rare1, rare2


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



def riemann_fwave_1d(hL, hR, huL, huR, bL, bR, uL, uR, phiL, phiR, s1, s2, g):
    num_eqn = 2
    num_waves = 2
    fw = np.zeros((num_eqn, num_waves))

    delh = hR - hL
    delhu = huR - huL
    delb = bR - bL
    delphidecomp = phiR - phiL + g * 0.5 * (hL + hR) * delb
    if (s2-s1)<10**(-3):
        beta1=0
        beta2=0
    else:
        beta1 = (s2 * delhu - delphidecomp) / (s2 - s1)
        beta2 = (delphidecomp - s1 * delhu) / (s2 - s1)

    # 1st nonlinear wave
    fw[0,0] = beta1
    fw[1,0] = beta1 * s1

    # 2nd nonlinear wave
    fw[0,1] = beta2
    fw[1,1] = beta2 * s2

    return fw


def barrier_passing(hL, hR, huL, huR, bL, bR, wall_height, drytol, g, maxiter):

    L2R = False
    R2L = False
    hstarL = 0.0
    hstarR = 0.0

    if (hL > drytol):
        uL = huL / hL
        hstar,_,_,_,_ = riemanntype(hL, hL, uL, -uL, maxiter, drytol, g)
        hstartest = max(hL, hstar)
        if (hstartest + bL > 0.5*(bL+bR)+wall_height):
            L2R = True
            hstarL = hstartest + bL - 0.5*(bL+bR) - wall_height

    if (hR > drytol):
        uR = huR / hR
        hstar,_,_,_,_ = riemanntype(hR, hR, -uR, uR, maxiter, drytol, g)
        hstartest = max(hR, hstar)
        if (hstartest + bR > 0.5*(bL+bR)+wall_height):
            R2L = True
            hstarR = hstartest + bR - 0.5*(bL+bR) - wall_height

    return L2R, R2L, hstarL, hstarR


def redistribute_fwave(q_l, q_r, aux_l, aux_r, wall_height, drytol, g, maxiter):

    fwave = np.zeros((2, 2, 2))
    s = np.zeros((2, 2))
    amdq = np.zeros((2, 2))
    apdq = np.zeros((2, 2))

    q_wall = np.zeros((2,3))
    aux_wall = np.zeros((1,3))
    s_wall = np.zeros(2)
    gamma = np.zeros((2,2))
    amdq_wall = np.zeros(2)
    apdq_wall = np.zeros(2)

    # hbox method
    q_wall[:,0] = q_l[:].copy()
    q_wall[:,2] = q_r[:].copy()

    aux_wall[0,0] = aux_l.copy()
    aux_wall[0,2] = aux_r.copy()
    aux_wall[0,1] = 0.5*(aux_wall[0,0] + aux_wall[0,2]) + wall_height

    L2R, R2L, hstarL, hstarR = barrier_passing(q_wall[0,0], q_wall[0,2], q_wall[1,0], q_wall[1,2], aux_wall[0,0], aux_wall[0,2], wall_height, drytol, g, maxiter)
    if (L2R==True and R2L==True):
        q_wall[0,1] = 0.5*(hstarL+hstarR)
        q_wall[1,1] = q_wall[0,1]  * (q_wall[1,0] + q_wall[1,2])/(q_wall[0,0] + q_wall[0,2]) # h*_avg * (huL+huR)/(hL+hR)

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
                hstar,_,_,_,_ = riemanntype(hL, hL, uL, -uL, maxiter, drytol, g)
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
                hstar,_,_,_,_ = riemanntype(hR, hR, -uR, uR, maxiter, drytol, g)
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
            fw = riemann_fwave_1d(hL, hR, huL, huR, bL, bR, uL, uR, phiL, phiR, s1, s2, g)
            s[0,i] = s1 * wall[0]
            s[1,i] = s2 * wall[1]
            fwave[:,0,i] = fw[:,0] * wall[0]
            fwave[:,1,i] = fw[:,1] * wall[1]
            # print("fw: ", fw)

            for mw in range(num_waves):
                if (s[mw,i] < 0):
                    amdq[:,i] += fwave[:,mw,i]
                elif (s[mw,i] > 0):
                    apdq[:,i] += fwave[:,mw,i]


    s_wall[0] = np.min(s)
    s_wall[1] = np.max(s)


    if s_wall[1] - s_wall[0] != 0.0:
        gamma[0,0] = (s_wall[1] * np.sum(fwave[0,:,:]) - np.sum(fwave[1,:,:])) / (s_wall[1] - s_wall[0])
        gamma[0,1] = (np.sum(fwave[1,:,:]) - s_wall[0] * np.sum(fwave[0,:,:])) / (s_wall[1] - s_wall[0])
        gamma[1,0] = gamma[0,0] * s_wall[0]
        gamma[1,1] = gamma[0,1] * s_wall[1]

    wave_wall = gamma
    # print("gamma[0,:]: ", gamma[0,:])
    for mw in range(2):
        if (s_wall[mw] < 0):
            amdq_wall[:] += gamma[:,mw]
        elif (s_wall[mw] > 0):
            apdq_wall[:] += gamma[:,mw]

    return wave_wall, s_wall, amdq_wall, apdq_wall



def shallow_fwave_dry_1d(q_l, q_r, aux_l, aux_r, problem_data):
    # print("shallow_fwave_hbox_dry_1d")
    g = problem_data['grav']
    drytol = problem_data['dry_tolerance']
    maxiter = problem_data['max_iteration']

    q_l.reshape((2,1))
    q_r.reshape((2,1))
    aux_l.reshape((1,1))
    aux_r.reshape((1,1))
    num_rp = 1 #q_l.shape[1]
    num_eqn = 2
    num_waves = 2
    num_ghost = 2


    # Output arrays
    fwave = np.zeros((num_eqn, num_waves, num_rp))
    s = np.zeros((num_waves, num_rp))
    amdq = np.zeros((num_eqn, num_rp))
    apdq = np.zeros((num_eqn, num_rp))

    for i in range(num_rp):
        hL = q_l[0]
        hR = q_r[0]
        huL = q_l[1]
        huR = q_r[1]
        bL = aux_l
        bR = aux_r

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
                hstar,_,_,_,_ = riemanntype(hL, hL, uL, -uL, maxiter, drytol, g)
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
                hstar,_,_,_,_ = riemanntype(hR, hR, -uR, uR, maxiter, drytol, g)
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

    # if problem_data['zero_width'] == True:
    #     nw = problem_data['wall_position']
    #     wall_height = problem_data['wall_height']
    #     iw = nw + num_ghost - 1
    #     fwave[:,:,iw], s[:,iw], amdq[:,iw], apdq[:,iw] = redistribute_fwave(q_l[:,iw:iw+1].copy(), q_r[:,iw:iw+1].copy(), aux_l[0,iw:iw+1].copy(), aux_r[0,iw:iw+1].copy(), wall_height, drytol, g, maxiter)


    return fwave, s, amdq, apdq

def wall_bc_flip(q,normal=1):
    """
    this function inverts the velocity across the normal direction and keeps the height the same
    """
    q_new = q
    q_new[normal] *= -1

    return q_new

def shallow_fwave_hbox_dry_1d(q_l, q_r, aux_l, aux_r, problem_data):
    # print("shallow_fwave_hbox_dry_1d")
    g = problem_data['grav']
    nw = problem_data['wall_position']
    wall_height = problem_data['wall_height']
    drytol = problem_data['dry_tolerance']
    maxiter = problem_data['max_iteration']
    alpha = problem_data['fraction']


    if False == False:
        MD = q_r[1,:]-q_l[1,:]
        num_rp = q_l.shape[1]
        num_eqn = 2
        num_waves = 2
        num_ghost = 2
        iw = nw + num_ghost -1

        # Output arrays
        fwave = np.zeros((num_eqn, num_waves, num_rp))
        s = np.zeros((num_waves, num_rp))
        amdq = np.zeros((num_eqn, num_rp))
        apdq = np.zeros((num_eqn, num_rp))

        # hboxes and ratios
        q_hbox_l = np.zeros((2,2))
        aux_hbox_l = np.zeros((1,2))
        q_hbox_l2 = np.zeros((2,2))
        aux_hbox_l2 = np.zeros((1,2))

        q_hbox_r = np.zeros((2,2))
        aux_hbox_r = np.zeros((1,2))
        q_hbox_r2 = np.zeros((2,2))
        aux_hbox_r2 = np.zeros((1,2))

        ratio1 = 2.0 * alpha / (1 + alpha)
        ratio2 = 2.0 * (1 - alpha) / (2 - alpha)

        # for the flux at left endpt boundary
        q_first = q_l[:,2]
        aux_first = aux_l[0,2]
        q_second = q_r[:,2]
        aux_second = aux_r[0,2]

        q_hbox_l[:,1] = alpha * q_first + (1-alpha)*q_second
        q_hbox_l[:,0] = wall_bc_flip(q_hbox_l[:,1])
        aux_hbox_l[0,0] = alpha * aux_first + (1-alpha) * aux_second
        aux_hbox_l[0,1] = alpha * aux_first + (1-alpha) * aux_second

        # for the BC flux
        #q_l[:,0] = q_hbox_l[:,1]

        # for the flux at the first cell edge off the left boundary
        q_hbox_l2[:,1] = q_second
        q_hbox_l2[:,0] = alpha * q_first + (1-alpha) * q_hbox_l[:,0]
        aux_hbox_l2[0,0] = alpha * aux_first + (1-alpha) * aux_hbox_l[0,0]
        aux_hbox_l2[0,1] = alpha * aux_first + (1-alpha) * aux_hbox_l[0,0]

        # for the flux at right endpt boundary
        q_last = q_r[:,-3]
        aux_last = aux_r[0,-3]
        q_penult = q_l[:,-3]
        aux_penult = aux_l[0,-3]

        q_hbox_r[:,0] = (alpha) * q_penult + (1-alpha) * q_last
        q_hbox_r[:,1] = wall_bc_flip(q_hbox_r[:,0])
        aux_hbox_r[0,0] = (alpha) * aux_penult + (1-alpha) * aux_last
        aux_hbox_r[0,1] = (alpha) * aux_penult + (1-alpha) * aux_last

        # for the BC flux
    #    q_r[:,1] = q_hbox_l[:,1]
    #    q_l[:,-2] = q_hbox_r[:,0]
        #q_r[:,-1] = q_hbox_r[:,0]

        # for the flux at the first cell edge off the right boundary
        q_hbox_r2[:,0] = q_penult
        q_hbox_r2[:,1] = (1-alpha)*q_last + alpha * q_hbox_r[:,1]
        aux_hbox_r2[0,0] = (1-alpha) * aux_last + alpha * aux_hbox_r[0,1]
        aux_hbox_r2[0,1] = (1-alpha) * aux_last + alpha * aux_hbox_r[0,1]


        # RP's
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
                    hstar,_,_,_,_ = riemanntype(hL, hL, uL, -uL, maxiter, drytol, g)
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
                    hstar,_,_,_,_ = riemanntype(hR, hR, -uR, uR, maxiter, drytol, g)
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

                fw = riemann_fwave_1d(hL, hR, huL, huR, bL, bR, uL, uR, phiL, phiR, s1, s2, g)
                s[0,i] = sL * wall[0]
                s[1,i] = sR * wall[1]
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
        fwave[:,:,[1]], s[:,[1]], amdq[:,[1]], apdq[:,[1]] = shallow_fwave_dry_1d(q_l[:,1],q_hbox_l[:,1],aux_l[0,1],aux_hbox_l[0,1],problem_data)
        fwave[:,:,[2]], s[:,[2]], amdq[:,[2]], apdq[:,[2]] = shallow_fwave_dry_1d(q_hbox_l2[:,0], q_hbox_l2[:,1], aux_hbox_l2[0,0], aux_hbox_l2[0,0], problem_data)
        fwave[:,:,[-3]], s[:,[-3]], amdq[:,[-3]], apdq[:,[-3]] = shallow_fwave_dry_1d(q_hbox_r2[:,0], q_hbox_r2[:,1], aux_hbox_r2[0,0], aux_hbox_r2[0,0], problem_data)
        fwave[:,:,[-2]], s[:,[-2]], amdq[:,[-2]], apdq[:,[-2]] = shallow_fwave_dry_1d(q_hbox_r[:,0],q_r[:,-2],aux_hbox_r[0,0],aux_r[0,-2],problem_data)
        fwave[:,:,iw], s[:,iw], amdq[:,iw], apdq[:,iw] = redistribute_fwave(q_l[:,iw].copy(), q_r[:,iw].copy(), aux_l[0,iw].copy(), aux_r[0,iw].copy(), wall_height, drytol, g, maxiter)

        #print("mass moemntum diff: ",(MD[:]-(amdq[0,:]+apdq[0,:])))#, " apdq+amdq: ", amdq[0,:]+apdq[0,:])

        return fwave, s, amdq, apdq
