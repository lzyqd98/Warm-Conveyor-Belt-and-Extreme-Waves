#%%
###### A simple PV model for reconstructing each wave event on an isentropic level ######
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt, timedelta as td
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
from matplotlib.collections import LineCollection
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import pickle
import glob
from netCDF4 import Dataset
import xarray as xr
import multiprocessing 
import argparse
parser = argparse.ArgumentParser(description="year for calculation dpvdt")
parser.add_argument("--year", type=int, required=True, help="year for calculation dpvdt")
args = parser.parse_args()
year = args.year

# The below package can be found from (https://github.com/paologhinassi/RWPtools), used to calculate local wave activity #
import lwa_pv_theta_cal
import wave_packet_cal
import lwa_pv_theta_addition
import lwa_pv_theta_addition2
import numpy.fft as _npfft

# monkey-patch the names inside LWA_additional
lwa_pv_theta_addition.fft    = _npfft.fft
lwa_pv_theta_addition.ifft   = _npfft.ifft
lwa_pv_theta_addition.zeros  = np.zeros
lwa_pv_theta_addition.append = np.append



#%%
def mean_lon_10(variable,axis):
    ri = -5
    variable_m = variable*0.0
    for i in np.arange(11):
        variable_m += np.roll(variable,int(ri+i), axis=axis)
    variable_m = variable_m/11.0
    return variable_m


def adv_divx(F):
    return (np.roll(F, -1, axis=1) - np.roll(F, 1, axis=1)) * inv_2dx

def adv_divy(F):
    dFdy = np.empty_like(F)
    dFdy[1:-1] = (F[2:] - F[:-2]) / (2 * dy)
    dFdy[0] = (F[1] - F[0]) / dy
    dFdy[-1] = (F[-1] - F[-2]) / dy
    return dFdy



import numpy as np
from scipy.fft import rfft, irfft, dct, idct

def spectral_hyperdiffuse_fft_dct(q, dt, dx_ref, dy_ref, p=3, tau=72*3600.0):
    """
    Apply exp(-nu * k^(2p) dt) in mixed spectral space (DCT in lat, rFFT in lon).

    Parameters
    ----------
    q : (nlat,nlon)
    dt : seconds
    dx_ref, dy_ref : meters (typical grid spacing where you care, e.g. 55e3)
    p : 1 for Laplacian, 3 for 6th-order
    tau : desired e-folding time at the *grid scale* (k ~ pi/dx_ref and pi/dy_ref)

    Returns
    -------
    q_new : (nlat,nlon)
    """
    nlat, nlon = q.shape

    # transforms
    A = dct(q, type=2, axis=0, norm='ortho')
    Q = rfft(A, axis=1)

    ky = np.arange(nlat)[:, None]
    kx = np.arange(Q.shape[1])[None, :]

    # convert mode index to physical wavenumber (approx)
    kx_phys = (2*np.pi*kx) / (nlon*dx_ref)   # rad/m
    ky_phys = (np.pi*ky) / (nlat*dy_ref)     # rad/m (DCT basis approx)

    k2 = kx_phys**2 + ky_phys**2

    # choose nu so that the highest resolved k has e-folding time tau
    # take k_max ~ pi/min(dx_ref,dy_ref)
    kmax = np.pi / min(dx_ref, dy_ref)
    nu = 1.0 / (tau * (kmax**(2*p)))

    # exponential (exact) damping for this linear operator
    damp = np.exp(-nu * (k2**p) * dt)
    Q *= damp

    # back transform
    A2 = irfft(Q, n=nlon, axis=1)
    q2 = idct(A2, type=2, axis=0, norm='ortho')
    return q2

import numpy as np
from scipy.linalg import solve_banded

def invert_pv_to_uv(qprime, dx, dy, Ld=np.inf):

    nlat, nlon = qprime.shape

    # Remove domain-mean vorticity anomaly
    qprime = qprime - np.nanmean(qprime)

    # Fourier transform ONLY in longitude
    qhat = np.fft.fft(qprime, axis=1)

    kx = 2*np.pi*np.fft.fftfreq(nlon, d=dx)

    psihat = np.zeros_like(qhat, dtype=complex)

    # Interior latitude points only
    ny = nlat - 2

    for m, k in enumerate(kx):

        if np.isinf(Ld):
            alpha = k**2
        else:
            alpha = k**2 + 1.0/Ld**2

        # d2psi/dy2 - alpha*psi = q
        main = (-2.0/dy**2 - alpha) * np.ones(ny)
        off  = (1.0/dy**2) * np.ones(ny-1)

        # Banded tridiagonal matrix
        ab = np.zeros((3, ny), dtype=complex)
        ab[0, 1:] = off
        ab[1, :]  = main
        ab[2, :-1] = off

        rhs = qhat[1:-1, m]

        psihat[1:-1, m] = solve_banded((1, 1), ab, rhs)

    # psi = 0 at meridional boundaries
    psihat[0, :]  = 0.0
    psihat[-1, :] = 0.0

    psi = np.real(np.fft.ifft(psihat, axis=1))

    # u = -dpsi/dy
    up = np.zeros_like(psi)
    up[1:-1] = -(psi[2:] - psi[:-2])/(2*dy)
    up[0]  = -(psi[1] - psi[0])/dy
    up[-1] = -(psi[-1] - psi[-2])/dy

    # v = dpsi/dx, periodic longitude
    vp = (
        np.roll(psi, -1, axis=1)
        - np.roll(psi, 1, axis=1)
    )/(2*dx)

    return up, vp


def qref_cal(pv, dm, lat,lon):
    ### This function is to calculate qref based on area analysis ###
    
    import numpy as np

    nlat = len(lat)   
    nlon = len(lon)       
    npart = nlat
    lat_mid = int(nlat/2) 
    
    # Create pv levels evenly, npart (nlat) levels in total
    qmax = np.nanmax(pv)
    qmin = np.nanmin(pv)
    qlevs = np.linspace(qmax, qmin, npart)   # qlev starts from maximum value to minimum value

    # Mass integral north of the latitude j
    lat_2d = lat[:, np.newaxis] * np.ones((nlat,nlon)) 
    M = np.zeros(nlat)  
    for j in np.arange(nlat):
        M[j] = np.sum(dm[lat_2d>= lat[j]])  

    # Mass integral north of the nth Q contour
    MQ = np.zeros(npart)
    for n in np.arange(npart):
        MQ[n] = np.sum(dm[pv >= qlevs[n]])  
    
    # Calculate qref via finding the mass equivalent between M and MQ
    qref = np.zeros(nlat)
    for j in np.arange(nlat-1):
        for n in np.arange(npart-1):
            if MQ[n]<=M[j] and MQ[n+1] > M[j]:
                r = (M[j]-MQ[n])/(MQ[n+1]-MQ[n])
                qref[j] = qlevs[n]*(1-r) + qlevs[n+1] * r
    
    qref[-1] = qmax

                
    return qref[lat_mid:]

def lwa_cal(a, pv, lon, lat, dmphi, dm):

    import numpy as np
    
    nlat = len(lat)
    nlon=len(lon)
    lat_mid = int(nlat/2)   
    nlat = len(lat)
    loo,laa = np.meshgrid(lon,lat)
    
    qref = np.zeros((nlat))
    qref[lat_mid:] = qref_cal(pv, dm, lat,lon)               # Northern hemisphere
    qref[lat_mid::-1] = -qref_cal(-pv[::-1,:], dm[::-1], lat,lon)  # Southern hemisphere

    lwa, lwa_a, lwa_c= lwa_integral(a, pv, qref, nlat, nlon, laa, lat, dmphi, dm)
                    
    return lwa, lwa_a, lwa_c


def lwa_integral(a, pv, qref, nlat, nlon, laa, lat, dmphi, dm):
    ### This function is to do the integration and output the final LWA, FAWA ###
    
    import numpy as np
    
    lwa = np.zeros((nlat,nlon))
    lwa_a = np.zeros((nlat,nlon))
    lwa_c = np.zeros((nlat,nlon))
    fawa = np.zeros((nlat))

    for j in np.arange(nlat):

        qboo = np.zeros((nlat,nlon))   
        qboo_a = np.zeros((nlat,nlon))
        qboo_c = np.zeros((nlat,nlon))
        
        qe = pv - qref[j]                    # The perturbation qe, difference between pv and qref for the kth latitude
        
        qboo[(laa<=lat[j]) & (qe>=0)] = 1               # PV increases with latitude, so we need areas where q>=0 but latitude <= lat[k]
        qboo[(laa>=lat[j]) & (qe<=0)] = -1              # And q<=0 but latitude >= lat[k]
        lwa[j,:] = np.sum(qboo * qe * dmphi, axis=0)    # LWA is the line integral

        qboo_a[(laa>=lat[j]) & (qe<=0)] = -1            # q<=0 and latitude >= lat[k] for anticyclonic component
        lwa_a[j,:] = np.sum(qboo_a * qe * dmphi, axis=0)    
        
        qboo_c[(laa<=lat[j]) & (qe>=0)] = 1             # q>=0 and latitude <= lat[k] for cyclonic component
        lwa_c[j,:] = np.sum(qboo_c * qe * dmphi, axis=0)    
        
    return lwa, lwa_a, lwa_c


#%%
def Reconstruct_pv(nn):
    
    ### This function is to directly interate the PV ###
    itertime = 10       # in minutes
    dt = itertime * 60  # in seconds
    total_time = 12
    nday = 2
    duration = nday * 4
    nsteps = int(duration*6*60/itertime) + 1  # number of iterate time steps over `duration` days

    eps_pv  = 1e-10
    f_res_cap   = 4e-5               # 1/s cap (tune)

    # read peaking date information
    peaking_date_index = Date.index(WE_peaking_date[nn])
    peaking_lon_index = np.squeeze(np.array(np.where(lon[:] == WE_peaking_lon[nn])))

    ### Read 6 hourly WCB forcings (dpvdt_wcb) ###
    F_wcb = dpvdt_wcb_all[nn][(total_time-duration):]
    Fz    = dpvdt_wcb_trans_all[nn][(total_time-duration):]
    
    ### Read PV, U, V, dPvdt_wcb and calculate dry forcings ###
    PV = np.zeros((duration, nlat, nlon))
    PV_up = np.zeros((duration, nlat, nlon))
    PV_down = np.zeros((duration, nlat, nlon))
    Rhotheta = np.zeros((duration, nlat, nlon))
    U = np.zeros((duration, nlat, nlon))
    V = np.zeros((duration, nlat, nlon))
    Fx = np.zeros((duration, nlat, nlon))
    Fy = np.zeros((duration, nlat, nlon))
    
    F_res = np.zeros((duration, nlat, nlon))
    F_total = np.zeros((duration, nlat, nlon))
    f_res = np.zeros((duration, nlat, nlon))
    
    for j in np.arange(duration):
        index = peaking_date_index - duration + j

        # read relevant terms at that time step #
        with xr.open_dataset(path[index]) as file:
            PV_d = file['PV'].values[isentrope_id, :, :]
            PV_d_down = file['PV'].values[isentrope_id-1, :, :]
            PV_d_up = file['PV'].values[isentrope_id+1, :, :]
            U_d  = file['U'].values[isentrope_id, :, :]
            V_d  = file['V'].values[isentrope_id, :, :]
            Rhotheta_d = file['Rhotheta'].values[isentrope_id, :, :]
            
        Fx_d = U_d * -adv_divx(PV_d)
        Fy_d = V_d * -adv_divy(PV_d)

        PV[j] =PV_d
        U[j] = U_d
        V[j] = V_d
        Rhotheta[j] = Rhotheta_d

        
        with xr.open_dataset(path[index+1]) as file_next:
            PV_d_next = file_next['PV'].values[isentrope_id, :, :]
            PV_d_down_next = file_next['PV'].values[isentrope_id-1, :, :]
            PV_d_up_next = file_next['PV'].values[isentrope_id+1, :, :]
            U_d_next  = file_next['U'].values[isentrope_id, :, :]
            V_d_next  = file_next['V'].values[isentrope_id, :, :]
            
        Fx_d_next = U_d_next * -adv_divx(PV_d_next)
        Fy_d_next = V_d_next * -adv_divy(PV_d_next)

                
        ### Calculate average forcing terms between two time steps ###
        PV_m = (PV_d + PV_d_next)/2.0
        PV_up_m = (PV_d_up+PV_d_up_next)/2.0
        PV_down_m = (PV_d_down+PV_d_down_next)/2.0
        PV_up[j] = PV_up_m
        PV_down[j] = PV_down_m
        
        F_total[j] = ( PV_d_next - PV_d ) / (6*3600)
        Fx[j] = (Fx_d + Fx_d_next) / 2.0
        Fy[j] = (Fy_d + Fy_d_next) / 2.0

        F_res[j] = ( PV_d_next - PV_d ) / (6*3600)  - Fx[j] - Fy[j] - F_wcb[j] - Fz[j]
                
    # Modify the dPvdt_wcb, remove PV tendency provided by WCBs #
    dlon1 = 60; dlon2 =-60
    F_wcb_mod = np.copy(F_wcb)
    F_wcb_roll = np.roll(F_wcb_mod, int(nlon/2)-peaking_lon_index, axis=2)
    F_wcb_roll[:,:, int(nlon/2)-int(dlon1/dlon):int(nlon/2)-int(dlon2/dlon)] = 0.0
    F_wcb_mod = np.roll(F_wcb_roll, peaking_lon_index-int(nlon/2), axis=2)

    # Modify the Fz, remove PV tendency provided by WCBs #
    dlon1 = 60; dlon2 =-60
    Fz_mod = np.copy(Fz)
    Fz_roll = np.roll(Fz_mod, int(nlon/2)-peaking_lon_index, axis=2)
    Fz_roll[:,:, int(nlon/2)-int(dlon1/dlon):int(nlon/2)-int(dlon2/dlon)] = 0.0
    Fz_mod = np.roll(Fz_roll, peaking_lon_index-int(nlon/2), axis=2)


    # Attach the peaking date data to the end of the arrays #
    with xr.open_dataset(path[peaking_date_index]) as file:
        PV_d = file['PV'].values[isentrope_id, :, :]
        U_d  = file['U'].values[isentrope_id, :, :]
        V_d  = file['V'].values[isentrope_id, :, :]
        Rhotheta_d = file['Rhotheta'].values[isentrope_id, :, :]
    PV = np.concatenate([PV, PV_d[np.newaxis, :, :]], axis=0)
    U = np.concatenate([U, U_d[np.newaxis, :, :]], axis=0)
    V = np.concatenate([V, V_d[np.newaxis, :, :]], axis=0)
    Rhotheta = np.concatenate([Rhotheta, Rhotheta_d[np.newaxis, :, :]], axis=0)

    ###------ Interpolate different forcing terms to higher resolution ------###
    PV_new   = np.zeros((nsteps, nlat, nlon))
    Rhotheta_new = np.zeros((nsteps, nlat, nlon))
    U_new   = np.zeros((nsteps, nlat, nlon)) 
    V_new   = np.zeros((nsteps, nlat, nlon)) 
    F_total_new = np.zeros((nsteps, nlat, nlon))
    
    F_res_new   = np.zeros((nsteps, nlat, nlon))
    f_res_new   = np.zeros((nsteps, nlat, nlon))
    F_wcb_new = np.zeros((nsteps, nlat, nlon))
    F_wcb_mod_new = np.zeros((nsteps, nlat, nlon))
    Fz_new = np.zeros((nsteps, nlat, nlon))
    Fz_mod_new = np.zeros((nsteps, nlat, nlon))

    # coarse snapshot times: 0, 1, 2, ..., duration-1   (each unit = 6 hours)
    t_edge = np.arange(duration+1)
    # coarse interval-midpoint times: 0.5, 1.5, 2.5, ..., duration-0.5
    t_mid = np.arange(duration) + 0.5
    # fine times in the same "6-hour unit"
    t_fine = np.linspace(0, duration, nsteps)


    for la in np.arange(nlat):
        for lo in np.arange(nlon):
            PV_new[:, la, lo] = np.interp(t_fine, t_edge, PV[:, la, lo])
            U_new[:, la, lo]  = np.interp(t_fine, t_edge, U[:, la, lo])
            V_new[:, la, lo]  = np.interp(t_fine, t_edge, V[:, la, lo])
            Rhotheta_new[:, la, lo] = np.interp(t_fine, t_edge, Rhotheta[:, la, lo])
            
    for la in np.arange(nlat):
        for lo in np.arange(nlon):
            F_total_new[:, la, lo]     = np.interp(t_fine, t_mid, F_total[:, la, lo])
            F_wcb_new[:, la, lo]       = np.interp(t_fine, t_mid, F_wcb[:, la, lo])
            F_wcb_mod_new[:, la, lo]   = np.interp(t_fine, t_mid, F_wcb_mod[:, la, lo])
            Fz_new[:, la, lo]          = np.interp(t_fine, t_mid, Fz[:, la, lo])
            Fz_mod_new[:, la, lo]      = np.interp(t_fine, t_mid, Fz_mod[:, la, lo])
            F_res_new[:, la, lo]       = np.interp(t_fine, t_mid, F_res[:, la, lo])
            f_res_new[:, la, lo]       = np.interp(t_fine, t_mid, f_res[:, la, lo])



    ###------ Iterate the initial value problem with RK3 method ------###
    q_full   = np.zeros((nsteps, nlat, nlon))
    q_full[0] = PV_new[0, :,:]
    q_noWCB = np.zeros((nsteps, nlat, nlon))    
    q_noWCB[0] = PV_new[0, :,:]
    
    # --- time stepping: SSPRK3 ---
    def RHS_full(q, j):
        ### without PV inversion ###
        Fx    = U_new[j,:,:] * -adv_divx(q)
        Fy    = V_new[j,:,:] * -adv_divy(q)
        F_res = F_res_new[j] 
        F_wcb = F_wcb_new[j]
        Fz    = Fz_new[j]
        return   Fx + Fy  + F_res + F_wcb + Fz

    

    def RHS_noWCB(q, qfull, j):
        ### without PV inversion ###
        Fx    = U_new[j,:,:] * -adv_divx(q)
        Fy    = V_new[j,:,:] * -adv_divy(q)
        F_res = F_res_new[j]       
        F_wcb = F_wcb_mod_new[j]
        Fz    = Fz_mod_new[j]

        
        return   Fx + Fy + F_res + F_wcb + Fz
    
        
    for j in range(nsteps-1):
        qn_full = q_full[j]
        qn_noWCB = q_noWCB[j]

        k1_full = RHS_full(qn_full, j)
        q1_full = qn_full + dt * k1_full
        q1_full = spectral_hyperdiffuse_fft_dct(q1_full, dt, dx_ref=8e4, dy_ref=1.1e5, p=3, tau=1*60)
        k1_noWCB = RHS_noWCB(qn_noWCB, qn_full, j)
        q1_noWCB = qn_noWCB + dt * k1_noWCB
        q1_noWCB = spectral_hyperdiffuse_fft_dct(q1_noWCB, dt, dx_ref=8e4, dy_ref=1.1e5, p=3, tau=1*60)


        k2_full = RHS_full(q1_full, j)
        q2_full = 0.75*qn_full + 0.25*(q1_full + dt*k2_full)
        q2_full = spectral_hyperdiffuse_fft_dct(q2_full, dt, dx_ref=8e4, dy_ref=1.1e5, p=3, tau=1*60)
        k2_noWCB = RHS_noWCB(q1_noWCB, q1_full, j)
        q2_noWCB = 0.75*qn_noWCB + 0.25*(q1_noWCB + dt*k2_noWCB)
        q2_noWCB = spectral_hyperdiffuse_fft_dct(q2_noWCB, dt, dx_ref=8e4, dy_ref=1.1e5, p=3, tau=1*60)


        k3_full = RHS_full(q2_full, j)
        q_full[j+1] = (1.0/3.0)*qn_full + (2.0/3.0)*(q2_full + dt*k3_full)
        q_full[j+1] = spectral_hyperdiffuse_fft_dct(q_full[j+1], dt, dx_ref=8e4, dy_ref=1.1e5, p=3, tau=1*60)
        k3_noWCB = RHS_noWCB(q2_noWCB, q2_full, j)
        q_noWCB[j+1] = (1.0/3.0)*qn_noWCB + (2.0/3.0)*(q2_noWCB + dt*k3_noWCB)
        q_noWCB[j+1] = spectral_hyperdiffuse_fft_dct(q_noWCB[j+1], dt, dx_ref=8e4, dy_ref=1.1e5, p=3, tau=1*60)


        
    PV_full = q_full
    PV_noWCB = q_noWCB
    
    ###------ Calculate lwa_wcb and dAdt_wcb------###
    PV_era5_arr = np.zeros((nday+1, nlat,nlon))
    PV_full_arr = np.zeros((nday+1, nlat,nlon))
    LWA_full_arr = np.zeros((nday+1, nlat,nlon))
    LWA_a_full_arr = np.zeros((nday+1, nlat,nlon))
    PV_noWCB_arr = np.zeros((nday+1, nlat,nlon))
    LWA_noWCB_arr = np.zeros((nday+1, nlat,nlon))
    LWA_a_noWCB_arr = np.zeros((nday+1, nlat,nlon))
    
    for d in np.arange(nday+1):
        file = xr.open_dataset("/scratch/bell/liu3315/ERA5/LWA_theta/"+Date[peaking_date_index-4*nday +d*4]+".nc")
        k = np.squeeze(np.where(file['theta_lev'].values == cross_level))
        rhotheta_i = file['Rhotheta'].values[k]
        qref = file['Qref'].values[k]
        va_i = file['V'].values[k]
        file.close()  
                    
        dm   = ds * rhotheta_i         # 2-D area differential element on isentrope (weighted by mass)
        dmphi = dphi_2d * rhotheta_i   # 2-D length differential element on isentrope (weighted by mass)
        
        lwa_full, lwa_a_full, lwa_c_full = lwa_cal(a, PV_full[d*144], lon, lat, dmphi, dm)
        lwa_noWCB, lwa_a_noWCB, lwa_c_noWCB = lwa_cal(a, PV_noWCB[d*144], lon, lat, dmphi, dm)

        ### Calculate LOCAL zonal wavenumber with wavelet analysis using meridional wind on isentropes ###
        k_1d = lwa_pv_theta_addition.zonalWN_fourier(va_i[np.newaxis,np.newaxis,:,:], lat, lon) 
            
        ### Calculate filtered LWA by Hann convolution, this is also the rossby wave packet ###               ### Smooth but not keep LWA conservation
        rwp_full = lwa_pv_theta_addition.Hann_convolution(lwa_full[np.newaxis,np.newaxis,:,:], lat, lon, k_1d, calibration = 1) ### Smooth and keep  LWA conservation
        rwp_full = rwp_full[0,0]
        rwp_noWCB = lwa_pv_theta_addition.Hann_convolution(lwa_noWCB[np.newaxis,np.newaxis,:,:], lat, lon, k_1d, calibration = 1) ### Smooth and keep  LWA conservation
        rwp_noWCB = rwp_noWCB[0,0]
        
        rwp_a_full = lwa_pv_theta_addition.Hann_convolution(lwa_a_full[np.newaxis,np.newaxis,:,:], lat, lon, k_1d, calibration = 1) ### Smooth and keep  LWA conservation
        rwp_a_full = rwp_a_full[0,0]
        rwp_a_noWCB = lwa_pv_theta_addition.Hann_convolution(lwa_a_noWCB[np.newaxis,np.newaxis,:,:], lat, lon, k_1d, calibration = 1) ### Smooth and keep  LWA conservation
        rwp_a_noWCB = rwp_a_noWCB[0,0]
        
        PV_full_arr[d] = PV_full[d*144]
        LWA_full_arr[d] = rwp_full
        LWA_a_full_arr[d] = rwp_a_full
        PV_noWCB_arr[d] = PV_noWCB[d*144]
        LWA_noWCB_arr[d] = rwp_noWCB
        LWA_a_noWCB_arr[d] = rwp_a_noWCB
        PV_era5_arr[d] = PV_new[d*144]
        
    lat_index = int(90 + WE_peaking_lat[nn])
    lon_index = int(WE_peaking_lon[nn])
    LWA_change = LWA_full_arr[-1][lat_index, lon_index] - LWA_noWCB_arr[-1][lat_index, lon_index]
    LWA_a_change = LWA_a_full_arr[-1][lat_index, lon_index] - LWA_a_noWCB_arr[-1][lat_index, lon_index]
        

    return PV_full_arr, PV_noWCB_arr, LWA_full_arr, LWA_a_full_arr, LWA_noWCB_arr, LWA_a_noWCB_arr, LWA_change, LWA_a_change, LWA_full_arr[-1][lat_index, lon_index], LWA_noWCB_arr[-1][lat_index, lon_index], LWA_a_full_arr[-1][lat_index, lon_index], LWA_a_noWCB_arr[-1][lat_index, lon_index]


#%%

if __name__ == "__main__":
    
    ###------ Constant ------###
    a= 6.378*1.e6               ## Earth radius 

    ###------ Parameters ------###
    variable_id = 'RWP'
    season_id = 'NDJFM'
    hemisphere_id = 'NH'
    isentrope_id = 2  ## 0:315K; 1:320K; 2:325K; 3:330K; 4:335K; 5:340K; 6:345K
    isentrope_name = '325K'
    cross_level = 325
        
    ###------ File paths for all dates ------###    
    path_all = glob.glob(r"/scratch/bell/liu3315/ERA5/LWA_theta/" + "*.nc")
    path_all.sort()
    path = [
        p for p in path_all
        if int(p.split("/")[-1][:4]) in [year, year+1]
    ]
    
    Date=[]
    for i in np.arange(len(path)):
        Date.append(path[i][-13:-3])
        
        
    ###------ Read lat/lon ------###
    file0 = xr.open_dataset(path[0])
    lat = file0['lat'].values
    lon = file0['lon'].values
    lat_mid = int(len(lat)/2)
    lat_SH = lat[0:lat_mid]
    lat_NH = lat[lat_mid:]
    nlon = len(lon)
    nlat = len(lat)
    nlat_SH = len(lat_SH)
    nlat_NH = len(lat_NH)
    dlat=lat[1]-lat[0]
    dlon=lon[1]-lon[0]
    dlon_rad = (lon[1] - lon[0]) * np.pi /180
    dlat_rad = (lat[1] - lat[0]) * np.pi /180
    file0.close()
    
    slat0   = np.sin(lat* np.pi/180)
    clat0   = np.cos(lat* np.pi/180)
    clat_2d = abs(clat0[:, np.newaxis] * np.ones((nlat,nlon)))     ## 2-D cos(fi) array
    slat_2d = slat0[:, np.newaxis] * np.ones((nlat,nlon))    
    
    dlamda =a * clat0[135] * dlon_rad                              ## IMPORTANT: use 45 degree dx for the entire domain (like a beta plane), avoid dealing with sphere problem
    dphi = a * dlat_rad   
    
    dphi_2d = a * dlat_rad * clat_2d                               ## 2-D length differential element: a*cos(fi)*dlat
    ds      = a**2 * clat_2d * dlat_rad * dlon_rad     
    
    inv_2dx = 1.0 / (2.0 * dlamda * np.ones((nlat,nlon)) )
    inv_dx2 = 1.0 / ((dlamda*np.ones((nlat,nlon)))**2)
    inv_dy2 = 1.0 / ((dphi* np.ones((nlat,nlon)))**2)
    dy = dphi
    dx = dlamda
    
    if hemisphere_id == 'NH':
        lat_hemi = lat_NH
    else:
        lat_hemi = lat_SH
            
    
    ###------ Read extreme events data ------###
    input_path = "/scratch/bell/liu3315/ERA5/LWA_theta/WE_theta/"+season_id+"/WE_"+hemisphere_id+"_"+isentrope_name+"_"+variable_id+"/" + str(year) +"/"
    
    with open(input_path+"dpvdt_wcb", "rb") as fp:             ## This is the diabatic heating term DPV/Dt estimated from each lagrangian WCB crossing
        dpvdt_wcb_all = pickle.load(fp)
    with open(input_path+"dpvdt_wcb_trans", "rb") as fp:       ## This is the vertical PV transport term
        dpvdt_wcb_trans_all = pickle.load(fp)

        
    with open(input_path+"WE_date", "rb") as fp:
        WE_date = pickle.load(fp)
    with open(input_path+"WE_peaking_date", "rb") as fp:
        WE_peaking_date = pickle.load(fp)        
    
    with open(input_path+"WE_lon", "rb") as fp:
        WE_lon = pickle.load(fp)
    with open(input_path+"WE_lat", "rb") as fp:
        WE_lat = pickle.load(fp)    

    with open(input_path+"WE_peaking_lon", "rb") as fp:
        WE_peaking_lon = pickle.load(fp)
    with open(input_path+"WE_peaking_lat", "rb") as fp:
        WE_peaking_lat = pickle.load(fp)   
                
    with open(input_path+"WE_LWA_max", "rb") as fp:
        WE_lwa_max = pickle.load(fp)
    with open(input_path+"WE_peaking_LWA_max", "rb") as fp:
        WE_peaking_lwa_max = pickle.load(fp)
            
    
    ###------ Parallel processing over events ------###
    numlist = np.arange(len(WE_peaking_date))
    numlist= numlist.astype(int)
    numlist = numlist.tolist()
    n_core=128
    pool = multiprocessing.Pool(processes=n_core)
    results = pool.map(Reconstruct_pv, numlist)
    pool.close()
    pool.join() 
    
    PV_full = np.array([res[0] for res in results])
    PV_noWCB = np.array([res[1] for res in results])
    LWA_full = np.array([res[2] for res in results])
    LWA_a_full = np.array([res[3] for res in results])
    LWA_noWCB = np.array([res[4] for res in results])
    LWA_a_noWCB = np.array([res[5] for res in results])
    LWA_change = np.array([res[6] for res in results])
    LWA_a_change = np.array([res[7] for res in results])
    LWA_full_peak = np.array([res[8] for res in results])
    LWA_noWCB_peak = np.array([res[9] for res in results])
    LWA_a_full_peak = np.array([res[10] for res in results])
    LWA_a_noWCB_peak = np.array([res[11] for res in results])

    np.save(input_path+"PV_full.npy", PV_full)
    np.save(input_path+"LWA_full", LWA_full)
    np.save(input_path+"LWA_a_full.npy", LWA_a_full)
    np.save(input_path+"PV_noWCB.npy", PV_noWCB)
    np.save(input_path+"LWA_noWCB", LWA_noWCB)
    np.save(input_path+"LWA_a_noWCB.npy", LWA_a_noWCB)
    np.save(input_path+"LWA_change_noWCB.npy", LWA_change)
    np.save(input_path+"LWA_a_change_noWCB.npy", LWA_a_change)
    np.save(input_path+"LWA_full_peak.npy", LWA_full_peak)
    np.save(input_path+"LWA_noWCB_peak.npy", LWA_noWCB_peak)
    np.save(input_path+"LWA_a_full_peak.npy", LWA_a_full_peak)
    np.save(input_path+"LWA_a_noWCB_peak.npy", LWA_a_noWCB_peak)


# %%
