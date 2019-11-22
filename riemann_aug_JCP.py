def riemann_aug_JCP(hL, hR, huL, huR, bL, bR, uL, uR, phiL, phiR, sE1, sE2, g):
    mwaves = 3
    delh = hR-hL
    delhu = huR-huL
    delphi = phiR-phiL
    delb = bR-bL
    #delP = pR - pL
    delnorm = delh**2 + delphi**2

    hm, s1m, s2m, rare1, rare2 = riemanntype(hL, hR, uL, uR, 1, drytol, g)
    sE1 = min(sE1,s2m) !Modified Einfeldt speed
    sE2 = max(sE2,s1m) !Modified Eindfeldt speed

    hstarHLL = max((huL-huR+sE2*hR-sE1*hL)/(sE2-sE1),0) # middle state in an HLL solve

    rarecorrectortest = False
    rarecorrector = False
    if rarecorrectortest == True:
       sdelta=sE2-sE1
       raremin = 0.5
       raremax = 0.9
       if rare1 == True and sE1*s1m < 0:
           raremin=0.2
       if rare2 == True and sE2*s2m < 0:
           raremin=0.2
       if rare1 == True or rare2 == True:
          #see which rarefaction is larger
          rare1st = 3 * (np.sqrt(g*hL)-np.sqrt(g*hm))
          rare2st= 3 * (np.sqrt(g*hR)-np.sqrt(g*hm))
          if max(rare1st,rare2st) > raremin*sdelta and max(rare1st,rare2st) < raremax*sdelta:
              rarecorrector = True
              if rare1st > rare2st:
                  lambda2 = s1m
              elif rare2st > rare1st:
                  lambda2 = s2m
              else:
                  lambda2=0.5*(s1m+s2m)
    if hstarHLL < min(hL,hR)/5:
        rarecorrector= False
    lamb = np.asarray([sE1, lambda2, sE3])
    r = np.zeros((3,mwaves))
    for mw in range(mwaves):
        r[0,mw]=1
        r[1,mw]=lamb[mw]
        r[2,mw]=(lamb[mw])**2

    if rarecorrector == False:
        lambda2 = 0.5*(lamb[1]+lamb[3])
#         lambda(2) = max(min(0.5d0*(s1m+s2m),sE2),sE1)
        r[0,1]=0
        r[1,1]=0
        r[2,1]=1
#     !determine the steady state wave -------------------
#      !criticaltol = 1.d-6
#      ! MODIFIED:
    criticaltol = max(drytol*g, 1e-6)
    criticaltol_2 = np.sqrt(criticaltol)
    deldelh = -delb
    deldelphi = -0.5d0 * (hR + hL) * (g * delb + 0 / rho)

#     !determine a few quanitites needed for steady state wave if iterated
    hLstar=hL
    hRstar=hR
    uLstar=uL
    uRstar=uR
    huLstar=uLstar*hLstar
    huRstar=uRstar*hRstar

    #!iterate to better determine the steady state wave
    convergencetol=1e-6
    for iter in range(maxiter):
    #!determine steady state wave (this will be subtracted from the delta vectors)
        if min(hLstar,hRstar) < drytol and rarecorrector==True:
            rarecorrector= False
            hLstar=hL
            hRstar=hR
            uLstar=uL
            uRstar=uR
            huLstar=uLstar*hLstar
            huRstar=uRstar*hRstar
            lambda2 = 0.5*(lamb[1]+lamb[3])
#           lambda(2) = max(min(0.5d0*(s1m+s2m),sE2),sE1)
            r[0,1]=0
            r[1,1]=0
            r[2,1]=1
        hbar =  max(0.5*(hLstar+hRstar),0)
        s1s2bar = 0.25*(uLstar+uRstar)**2 - g*hbar
        s1s2tilde= max(0,uLstar*uRstar) - g*hbar

        sonic = False
        if abs(s1s2bar) <= criticaltol:
            sonic = True
        elif s1s2bar*s1s2tilde <= criticaltol**2:
            sonic = True
        elif s1s2bar*sE1*sE2 <= criticaltol**2:
            sonic = True
        elif min(abs(sE1),abs(sE2)) < criticaltol_2:
            sonic = True
        elif sE1 <  criticaltol_2 and s1m > -criticaltol_2:
            sonic = True
        elif sE2 > -criticaltol_2 and s2m <  criticaltol_2:
            sonic = True
        elif (uL+np.sqrt(g*hL))*(uR+np.sqrt(g*hR)) < 0:
            sonic = True
        elif (uL- np.sqrt(g*hL))*(uR- np.sqrt(g*hR)) < 0:
            sonic = True

#        !find jump in h, deldelh
        if sonic==True:
            deldelh =  -delb
        else:
            deldelh = delb*g*hbar/s1s2bar
#        !find bounds in case of critical state resonance, or negative states
        if sE1 < -criticaltol and sE2 > criticaltol:
            deldelh = min(deldelh,hstarHLL*(sE2-sE1)/sE2)
            deldelh = max(deldelh,hstarHLL*(sE2-sE1)/sE1)
        elif sE1 >= criticaltol:
            deldelh = min(deldelh,hstarHLL*(sE2-sE1)/sE1)
            deldelh = max(deldelh,-hL)
        elif sE2 < -criticaltol:
            deldelh = min(deldelh,hR)
            deldelh = max(deldelh,hstarHLL*(sE2-sE1)/sE2)

#        ! adjust deldelh for well-balancing of atmospheric pressure difference
    #    deldelh = deldelh - delP/(rho*g)

#        !find jump in phi, deldelphi
        if sonic == True:
            deldelphi = -g*hbar*delb
        else:
            deldelphi = -delb*g*hbar*s1s2tilde/s1s2bar
#        !find bounds in case of critical state resonance, or negative states
        deldelphi=min(deldelphi,g*max(-hLstar*delb,-hRstar*delb))
        deldelphi=max(deldelphi,g*min(-hLstar*delb,-hRstar*delb))
        deldelphi = deldelphi # - hbar * delp / rho

        delt = np.zeros(3)
        delt[0]=delh-deldelh
        delt[1]=delhu
        delt[2]=delphi-deldelphi

#        !Determine determinant of eigenvector matrix========
        det1=r[0,0]*(r[1,1]*r[2,2]-r[1,2]*r[2,1])
        det2=r[0,1]*(r[1,0]*r[2,2]-r[1,2]*r[2,0])
        det3=r[0,2]*(r[1,0]*r[2,1]-r[1,1]*r[2,0])
        determinant=det1-det2+det3


#        !solve for beta(k) using Cramers Rule=================
        A = np.zeros((3,3))
        beta = np.zeros(3)
        for k in range(3):
            for mw in range(3):
                A[0,mw]=r[0,mw]
                A[1,mw]=r[1,mw]
                A[2,mw]=r[2,mw]

            A[0,k]=delt[0]
            A[1,k]=delt[1]
            A[2,k]=delt[2]
            det1=A[0,0]*(A[1,1]*A[2,2]-A[1,2]*A[2,1])
            det2=A[0,1]*(A[1,0]*A[2,2]-A[1,2]*A[2,0])
            det3=A[0,2]*(A[1,0]*A[2,1]-A[1,1]*A[2,0])
            beta[k]=(det1-det2+det3)/determinant

        if abs(delt[0]**2+delt[2]**2-delnorm) < convergencetol:
            break
        delnorm = delt[0]**2+delt[2]**2
        # !find new states qLstar and qRstar on either side of interface
        hLstar=hL
        hRstar=hR
        uLstar=uL
        uRstar=uR
        huLstar=uLstar*hLstar
        huRstar=uRstar*hRstar
        for mw in range(mwaves):
            if lamb[mw] < 0:
                hLstar= hLstar + beta[mw]*r[0,mw]
                huLstar= huLstar + beta[mw]*r[1,mw]
        for mw in range(mwaves): # mw=mwaves,1,-1
            if lamb[mwaves-mw-1] > 0:
                hRstar= hRstar - beta[mw]*r[0,mw]
                huRstar= huRstar - beta[mw]*r[1,mw]

        if hLstar > drytol:
            uLstar=huLstar/hLstar
        else:
            hLstar=max(hLstar,0)
            uLstar=0
        if hRstar > drytol:
            uRstar=huRstar/hRstar
        else:
            hRstar=max(hRstar,0)
            uRstar=0

#      enddo ! end iteration on Riemann problem
    sw = np.zeros(mwaves)
    fw = np.zeros((3,mwaves))
    for mw in range(mwaves):
        sw[mw]=lamb[mw]
        fw[0,mw]=beta[mw]*r[1,mw]
        fw[1,mw]=beta[mw]*r[2,mw]
        fw[2,mw]=beta[mw]*r[1,mw]
#      !find transverse components (ie huv jumps).
      # ! MODIFIED from 5.3.1 version
      # fw(3,1)=fw(3,1)*vL
      # fw(3,3)=fw(3,3)*vR
      # fw(3,2)= 0.d0

    # hustar_interface = huL + fw[0,0]  #  = huR - fw(1,3)
    # if hustar_interface <= 0.0:
    #     fw[2,0] = fw[2,0] + (hR*uR*vR - hL*uL*vL - fw[2,0]- fw[2,2])
    # else:
    #     fw[2,2] = fw[2,2] + (hR*uR*vR - hL*uL*vL - fw[2,0]- fw[2,2])

    return fw # note that this is fw for [h,hu,b]
##################################################################################################################
#       subroutine riemann_aug_JCP(maxiter,meqn,mwaves,hL,hR,huL,huR,
#      &   hvL,hvR,bL,bR,uL,uR,vL,vR,phiL,phiR,pL,pR,sE1,sE2,drytol,g,rho,
#      &   sw,fw)
#
#       ! solve shallow water equations given single left and right states
#       ! This solver is described in J. Comput. Phys. (6): 3089-3113, March 2008
#       ! Augmented Riemann Solvers for the Shallow Equations with Steady States and Inundation
#
#       ! To use the original solver call with maxiter=1.
#
#       ! This solver allows iteration when maxiter > 1. The iteration seems to help with
#       ! instabilities that arise (with any solver) as flow becomes transcritical over variable topo
#       ! due to loss of hyperbolicity.
#
#       implicit none
#
#       !input
#       integer meqn,mwaves,maxiter
#       double precision fw(meqn,mwaves)
#       double precision sw(mwaves)
#       double precision hL,hR,huL,huR,bL,bR,uL,uR,phiL,phiR,sE1,sE2
#       double precision hvL,hvR,vL,vR,pL,pR
#       double precision drytol,g,rho
#
#
#       !local
#       integer m,mw,k,iter
#       double precision A(3,3)
#       double precision r(3,3)
#       double precision lambda(3)
#       double precision del(3)
#       double precision beta(3)
#
#       double precision delh,delhu,delphi,delb,delnorm
#       double precision rare1st,rare2st,sdelta,raremin,raremax
#       double precision criticaltol,convergencetol,raretol
#       double precision criticaltol_2, hustar_interface
#       double precision s1s2bar,s1s2tilde,hbar,hLstar,hRstar,hustar
#       double precision huRstar,huLstar,uRstar,uLstar,hstarHLL
#       double precision deldelh,deldelphi,delP
#       double precision s1m,s2m,hm
#       double precision det1,det2,det3,determinant
#
#       logical rare1,rare2,rarecorrector,rarecorrectortest,sonic
#
#       !determine del vectors
#       delh = hR-hL
#       delhu = huR-huL
#       delphi = phiR-phiL
#       delb = bR-bL
#       delP = pR - pL
#       delnorm = delh**2 + delphi**2
#
#       call riemanntype(hL,hR,uL,uR,hm,s1m,s2m,rare1,rare2,
#      &                                          1,drytol,g)
#
#
#       lambda(1)= min(sE1,s2m) !Modified Einfeldt speed
#       lambda(3)= max(sE2,s1m) !Modified Eindfeldt speed
#       sE1=lambda(1)
#       sE2=lambda(3)
#       lambda(2) = 0.d0  ! ### Fix to avoid uninitialized value in loop on mw -- Correct?? ###
#
#
#       hstarHLL = max((huL-huR+sE2*hR-sE1*hL)/(sE2-sE1),0.d0) ! middle state in an HLL solve
#
# c     !determine the middle entropy corrector wave------------------------
#       rarecorrectortest=.false.
#       rarecorrector=.false.
#       if (rarecorrectortest) then
#          sdelta=lambda(3)-lambda(1)
#          raremin = 0.5d0
#          raremax = 0.9d0
#          if (rare1.and.sE1*s1m.lt.0.d0) raremin=0.2d0
#          if (rare2.and.sE2*s2m.lt.0.d0) raremin=0.2d0
#          if (rare1.or.rare2) then
#             !see which rarefaction is larger
#             rare1st=3.d0*(sqrt(g*hL)-sqrt(g*hm))
#             rare2st=3.d0*(sqrt(g*hR)-sqrt(g*hm))
#             if (max(rare1st,rare2st).gt.raremin*sdelta.and.
#      &         max(rare1st,rare2st).lt.raremax*sdelta) then
#                   rarecorrector=.true.
#                if (rare1st.gt.rare2st) then
#                   lambda(2)=s1m
#                elseif (rare2st.gt.rare1st) then
#                   lambda(2)=s2m
#                else
#                   lambda(2)=0.5d0*(s1m+s2m)
#                endif
#             endif
#          endif
#          if (hstarHLL.lt.min(hL,hR)/5.d0) rarecorrector=.false.
#       endif
#
# c     ## Is this correct 2-wave when rarecorrector == .true. ??
#       do mw=1,mwaves
#          r(1,mw)=1.d0
#          r(2,mw)=lambda(mw)
#          r(3,mw)=(lambda(mw))**2
#       enddo
#       if (.not.rarecorrector) then
#          lambda(2) = 0.5d0*(lambda(1)+lambda(3))
# c         lambda(2) = max(min(0.5d0*(s1m+s2m),sE2),sE1)
#          r(1,2)=0.d0
#          r(2,2)=0.d0
#          r(3,2)=1.d0
#       endif
# c     !---------------------------------------------------
#
#
# c     !determine the steady state wave -------------------
#       !criticaltol = 1.d-6
#       ! MODIFIED:
#       criticaltol = max(drytol*g, 1d-6)
#       criticaltol_2 = sqrt(criticaltol)
#       deldelh = -delb
#       deldelphi = -0.5d0 * (hR + hL) * (g * delb + delp / rho)
#
# c     !determine a few quanitites needed for steady state wave if iterated
#       hLstar=hL
#       hRstar=hR
#       uLstar=uL
#       uRstar=uR
#       huLstar=uLstar*hLstar
#       huRstar=uRstar*hRstar
#
#       !iterate to better determine the steady state wave
#       convergencetol=1.d-6
#       do iter=1,maxiter
#          !determine steady state wave (this will be subtracted from the delta vectors)
#          if (min(hLstar,hRstar).lt.drytol.and.rarecorrector) then
#             rarecorrector=.false.
#             hLstar=hL
#             hRstar=hR
#             uLstar=uL
#             uRstar=uR
#             huLstar=uLstar*hLstar
#             huRstar=uRstar*hRstar
#             lambda(2) = 0.5d0*(lambda(1)+lambda(3))
# c           lambda(2) = max(min(0.5d0*(s1m+s2m),sE2),sE1)
#             r(1,2)=0.d0
#             r(2,2)=0.d0
#             r(3,2)=1.d0
#          endif
#
#          hbar =  max(0.5d0*(hLstar+hRstar),0.d0)
#          s1s2bar = 0.25d0*(uLstar+uRstar)**2 - g*hbar
#          s1s2tilde= max(0.d0,uLstar*uRstar) - g*hbar
#
# c        !find if sonic problem
#          ! MODIFIED from 5.3.1 version
#          sonic = .false.
#          if (abs(s1s2bar) <= criticaltol) then
#             sonic = .true.
#          else if (s1s2bar*s1s2tilde <= criticaltol**2) then
#             sonic = .true.
#          else if (s1s2bar*sE1*sE2 <= criticaltol**2) then
#             sonic = .true.
#          else if (min(abs(sE1),abs(sE2)) < criticaltol_2) then
#             sonic = .true.
#          else if (sE1 <  criticaltol_2 .and. s1m > -criticaltol_2) then
#             sonic = .true.
#          else if (sE2 > -criticaltol_2 .and. s2m <  criticaltol_2) then
#             sonic = .true.
#          else if ((uL+dsqrt(g*hL))*(uR+dsqrt(g*hR)) < 0.d0) then
#             sonic = .true.
#          else if ((uL- dsqrt(g*hL))*(uR- dsqrt(g*hR)) < 0.d0) then
#             sonic = .true.
#          end if
#
# c        !find jump in h, deldelh
#          if (sonic) then
#             deldelh =  -delb
#          else
#             deldelh = delb*g*hbar/s1s2bar
#          endif
# c        !find bounds in case of critical state resonance, or negative states
#          if (sE1.lt.-criticaltol.and.sE2.gt.criticaltol) then
#             deldelh = min(deldelh,hstarHLL*(sE2-sE1)/sE2)
#             deldelh = max(deldelh,hstarHLL*(sE2-sE1)/sE1)
#          elseif (sE1.ge.criticaltol) then
#             deldelh = min(deldelh,hstarHLL*(sE2-sE1)/sE1)
#             deldelh = max(deldelh,-hL)
#          elseif (sE2.le.-criticaltol) then
#             deldelh = min(deldelh,hR)
#             deldelh = max(deldelh,hstarHLL*(sE2-sE1)/sE2)
#          endif
#
# c        ! adjust deldelh for well-balancing of atmospheric pressure difference
#          deldelh = deldelh - delP/(rho*g)
#
# c        !find jump in phi, deldelphi
#          if (sonic) then
#             deldelphi = -g*hbar*delb
#          else
#             deldelphi = -delb*g*hbar*s1s2tilde/s1s2bar
#          endif
# c        !find bounds in case of critical state resonance, or negative states
#          deldelphi=min(deldelphi,g*max(-hLstar*delb,-hRstar*delb))
#          deldelphi=max(deldelphi,g*min(-hLstar*delb,-hRstar*delb))
#          deldelphi = deldelphi - hbar * delp / rho
#
#          del(1)=delh-deldelh
#          del(2)=delhu
#          del(3)=delphi-deldelphi
#
# c        !Determine determinant of eigenvector matrix========
#          det1=r(1,1)*(r(2,2)*r(3,3)-r(2,3)*r(3,2))
#          det2=r(1,2)*(r(2,1)*r(3,3)-r(2,3)*r(3,1))
#          det3=r(1,3)*(r(2,1)*r(3,2)-r(2,2)*r(3,1))
#          determinant=det1-det2+det3
#
# c        !solve for beta(k) using Cramers Rule=================
#          do k=1,3
#             do mw=1,3
#                   A(1,mw)=r(1,mw)
#                   A(2,mw)=r(2,mw)
#                   A(3,mw)=r(3,mw)
#             enddo
#             A(1,k)=del(1)
#             A(2,k)=del(2)
#             A(3,k)=del(3)
#             det1=A(1,1)*(A(2,2)*A(3,3)-A(2,3)*A(3,2))
#             det2=A(1,2)*(A(2,1)*A(3,3)-A(2,3)*A(3,1))
#             det3=A(1,3)*(A(2,1)*A(3,2)-A(2,2)*A(3,1))
#             beta(k)=(det1-det2+det3)/determinant
#          enddo
#
#          !exit if things aren't changing
#          if (abs(del(1)**2+del(3)**2-delnorm).lt.convergencetol) exit
#          delnorm = del(1)**2+del(3)**2
#          !find new states qLstar and qRstar on either side of interface
#          hLstar=hL
#          hRstar=hR
#          uLstar=uL
#          uRstar=uR
#          huLstar=uLstar*hLstar
#          huRstar=uRstar*hRstar
#          do mw=1,mwaves
#             if (lambda(mw).lt.0.d0) then
#                hLstar= hLstar + beta(mw)*r(1,mw)
#                huLstar= huLstar + beta(mw)*r(2,mw)
#             endif
#          enddo
#          do mw=mwaves,1,-1
#             if (lambda(mw).gt.0.d0) then
#                hRstar= hRstar - beta(mw)*r(1,mw)
#                huRstar= huRstar - beta(mw)*r(2,mw)
#             endif
#          enddo
#
#          if (hLstar.gt.drytol) then
#             uLstar=huLstar/hLstar
#          else
#             hLstar=max(hLstar,0.d0)
#             uLstar=0.d0
#          endif
#          if (hRstar.gt.drytol) then
#             uRstar=huRstar/hRstar
#          else
#             hRstar=max(hRstar,0.d0)
#             uRstar=0.d0
#          endif
#
#       enddo ! end iteration on Riemann problem
#
#       do mw=1,mwaves
#          sw(mw)=lambda(mw)
#          fw(1,mw)=beta(mw)*r(2,mw)
#          fw(2,mw)=beta(mw)*r(3,mw)
#          fw(3,mw)=beta(mw)*r(2,mw)
#       enddo
#       !find transverse components (ie huv jumps).
#       ! MODIFIED from 5.3.1 version
#       fw(3,1)=fw(3,1)*vL
#       fw(3,3)=fw(3,3)*vR
#       fw(3,2)= 0.d0
#
#       hustar_interface = huL + fw(1,1)   ! = huR - fw(1,3)
#       if (hustar_interface <= 0.0d0) then
#           fw(3,1) = fw(3,1) + (hR*uR*vR - hL*uL*vL - fw(3,1)- fw(3,3))
#         else
#           fw(3,3) = fw(3,3) + (hR*uR*vR - hL*uL*vL - fw(3,1)- fw(3,3))
#         end if
#
#
#       return
#
#       end !subroutine riemann_aug_JCP-------------------------------------------------
