#%%
###### This code is to track extreme wave events in CMIP6 ######
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import datetime as dt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import pandas as pd
import cv2
import copy
import matplotlib.path as mpath
import pickle
import glob
from netCDF4 import Dataset
import os
import xarray as xr
import multiprocessing ### Parallel Coding ###


#%%
### A function to calculate distance between two grid points on earth ###
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2): # longitude1，latitude1，longitude2，latitude2 （in degrees）
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # degress to rad
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    # haversine equation 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6378 # earth radius
    return c * r * 1000


def WE_track(nyear):
    
    ###------ Select files for a specific season and years ------### 
    path_year = [
        f for f in path
        if (f[-13:-9] == str(nyear) and f[-9:-7] in ("11","12")) or
        (f[-13:-9] == str(nyear+1) and f[-9:-7] in ("01", "02","03"))
    ]
    
    # path_year = [
    #     f for f in path
    #     if (f[-13:-9] == str(nyear) and f[-9:-7] in ("06", "07", "08")) 
    # ]
    
    # path_year = [
    #     f for f in path
    #     if (f[-13:-9] == str(nyear) and f[-9:-7] in ("07","08", "09","10","11", "12")) or
    #     (f[-13:-9] == str(nyear+1) and f[-9:-7] in ("01", "02", "03","04", "05", "06"))
    # ]
    
    Date_year=[]
    for i in np.arange(len(path_year)):
        Date_year.append(path_year[i][-13:-3])
    
    ###------ Read basic variables ------###
    path0 = path_year[0]
    file0 = Dataset(path0,'r')
    lon = file0.variables['lon'][:]
    lat = file0.variables['lat'][:]
    lat_mid = int(len(lat)/2)
    lat_SH = lat[0:lat_mid]
    lat_NH = lat[lat_mid:]
    nlon = len(lon)
    nlat = len(lat)
    nlat_SH = len(lat_SH)
    nlat_NH = len(lat_NH)
    file0.close()
    
    a= 6.378*1.e6    ### Earth radius in meters
    slat0   = np.sin(lat*pi/180)
    clat0   = np.cos(lat*pi/180)
    clat_2d = abs(clat0[:, np.newaxis] * np.ones((nlat,nlon)))                    ## 2-D cos(fi) array
    slat_2d = slat0[:, np.newaxis] * np.ones((nlat,nlon))                         ## 2-D sin(fi) array
    dlat = (lat[1] - lat[0]) * pi /180                                            ## latitude spacing
    dlon = (lon[1] - lon[0]) * pi /180                                            ## longitude spacing
    dphi = a * dlat * clat_2d                                                     ## 2-D length differential element: a*cos(fi)*dlat
    ds   = a**2 * clat_2d * dlat * dlon                                           ## 2-D area differential element:   a^2*cos(fi)*dlat*dlon
    ds_NH = ds[lat_mid:,:] 
    ds_SH = ds[0:lat_mid,:]


    ###------ Read LWA or RWP data (Isentrope: 320, 325, 330, 335K) ------###
    file_path = "/scratch/bell/liu3315/ERA5/LWA_theta/LWA_theta_combined/"+season_id+"/"+variable_id1+"_"+season_id+"_"+ str(nyear) + "_"+str(nyear+1)+".nc"
    file = Dataset(file_path, "r")
    if hemisphere_id == 'SH':
        var = file.variables[variable_id2][:, isentrope_id,0:lat_mid,:]
    else:
        var = file.variables[variable_id2][:, isentrope_id,lat_mid:,]
    file.close()
    nday = len(var)


    #%%
    ###### Detect extreme wave event ######
    if hemisphere_id == 'SH':
        nlat_hemi = nlat_SH
        lat_hemi=lat_SH
        ds_hemi = ds_SH
    else:
        nlat_hemi = nlat_NH
        lat_hemi=lat_NH
        ds_hemi = ds_NH

    Extreme_freq = np.zeros((nlat_hemi, nlon))         ### The final ExtremeWE frequency/total number 

    ### ------ Threshold of wave events (Default is the median of 6houlry maximum LWA ) ------###    
    var_max_lon = np.zeros((nlon*nday))
    for t in np.arange(nday):    
        for lo in np.arange(nlon):
            var_max_lon[t*nlon+lo] = np.max(var[t,:,lo])
    Thresh = np.percentile(var_max_lon[:],50)
    # Thresh = 60 
    
    
    ###------ Connected component-labeling algorithm (CV2) ------###    
    WE = np.zeros((nday,nlat_hemi,nlon),dtype='int8') 
    WE[var>Thresh] = 255                    # Wave event
    num_labels = np.zeros(nday)
    labels = np.zeros((nday,nlat_hemi,nlon))
    for d in np.arange(nday):
        num_labels[d], labels[d,:,:], stats, centroids  = cv2.connectedComponentsWithStats(WE[d,:,:], connectivity=4)

    ###------ Connect labels across 0/360 ------###
    labels_new = copy.copy(labels)
    for d in np.arange(nday):
        if np.any(labels_new[d,:,0]) == 0 or np.any(labels_new[d,:,-1]) == 0:   ## If there are no events at the left/right edge, then we don't need to do anything
            continue
                
        column_0 = np.zeros((nlat_hemi,3))       ## We assume there are at most three wave events at the edges (actually most of time there is just one)
        column_end = np.zeros((nlat_hemi,3))
        label_0 = np.zeros(3)
        label_end = np.zeros(3)
        
        ## Get the wave event at left edge ##
        start_lat0 = 0
        for i in np.arange(3):
            for la in np.arange(start_lat0, nlat_hemi):
                if labels_new[d,la,0]==0:
                    continue
                if labels_new[d,la,0]!=0:
                    label_0[i]=labels_new[d,la,0]
                    column_0[la,i]=labels_new[d,la,0]
                    if labels_new[d,la+1,0]!=0:
                        continue
                    if labels_new[d,la+1,0]==0:
                        start_lat0 = la+1
                        break 

            ## Get the wave event at right edge ## 
            start_lat1 = 0
            for j in np.arange(3):
                for la in np.arange(start_lat1, nlat_hemi):
                    if labels_new[d,la,-1]==0:
                        continue
                    if labels_new[d,la,-1]!=0:
                        label_end[j]=labels_new[d,la,-1]
                        column_end[la,j]=labels_new[d,la,-1]
                        if labels_new[d,la+1,-1]!=0:
                            continue
                        if labels_new[d,la+1,-1]==0:
                            start_lat1 = la+1
                            break                       
                ## Compare the two cloumns at right and left edges, and connect the label if the two are indeed connected
                if (column_end[:,i]*column_0[:,j]).mean() == 0:
                    continue                
                if (column_end*column_0).mean() != 0:
                    num_labels[d]-=1
                    if label_0[i] < label_end[j]:
                        labels_new[d][labels_new[d]==label_end[j]] = label_0[i]
                        labels_new[d][labels_new[d]>label_end[j]] = (labels_new[d]-1)[labels_new[d]>label_end[j]]            
                    if label_0[i] > label_end[j]:
                        labels_new[d][labels_new[d]==label_0[i]] = label_end[j]
                        labels_new[d][labels_new[d]>label_0[i]] = (labels_new[d]-1)[labels_new[d]>label_0[i]]


    ### ------ Now we get different information of each individule wave patch ------###
    lat_d = []; lon_d = []; lwa_max=[]; lwa_total =[]; lwa_mean = []; lwa_label=[]
    lon_w = []; lon_e = []; area = []
    lat_n = []; lat_s = []
    lon_wide = []; lat_wide = []
    for d in np.arange(nday):
        if int(num_labels[d]-1)==0:
            lat_d.append([np.nan])
            lon_d.append([np.nan])
            lon_w.append([np.nan])
            lon_e.append([np.nan])
            lat_n.append([np.nan])
            lat_s.append([np.nan])
            area.append([np.nan])
            lon_wide.append([np.nan])
            lat_wide.append([np.nan])
            lwa_max.append([np.nan])
            lwa_total.append([np.nan])
            lwa_mean.append([np.nan])
            lwa_label.append([np.nan])
            continue
        
        lat_list=[];    lon_list=[];   lwa_max_list=[]; lwa_total_list = []; lwa_mean_list = []; lwa_label_list = []
        lon_w_list = [];lon_e_list=[]; area_list = []
        lon_wide_list = []
        lat_n_list = []; lat_s_list=[]
        lat_wide_list = []
        
        for n in np.arange(0,int(num_labels[d]-1)):
            LWA_d = np.zeros((nlat_hemi, nlon))
            LWA_d[labels_new[d]==n+1]=var[d][labels_new[d]==n+1]   ### isolate that wave event ###
            
                
            ###------ Get the maximum LWA location ------###
            if len(np.array(np.where( LWA_d==LWA_d.max() ))[0])>1:           
                lat_list.append( lat_hemi[np.squeeze(np.array(np.where( LWA_d==LWA_d.max() )))[0][0]])
                lon_list.append( lon[np.squeeze(np.array(np.where( LWA_d==LWA_d.max() )))[1][0]])
                lwa_max_list.append( LWA_d.max())
            else:
                lat_list.append( lat_hemi[np.squeeze(np.array(np.where( LWA_d==LWA_d.max() )))[0]])
                lon_list.append( lon[np.squeeze(np.array(np.where( LWA_d==LWA_d.max() )))[1]])
                lwa_max_list.append( LWA_d.max())
                
            ###------- Get the total LWA ------###
            lwa_total_list.append(LWA_d[labels_new[d]==n+1].sum())
            
            ###------ Get the mean LWA ------###
            lwa_mean_list.append(np.mean(LWA_d[labels_new[d]==n+1]))
            
            ###------ Get all LWA of this event ------###
            lwa_label_list.append(LWA_d)
            
            ###------ Get the width from west to east ------###            
            for lo in np.arange(nlon):
                if (np.any(LWA_d[:,lo])) and (not np.any(LWA_d[:,lo-1])):
                    lon_w_list.append(lon[lo]) 
                if (not np.any(LWA_d[:,lo])) and (np.any(LWA_d[:,lo-1])):
                    lon_e_list.append(lon[lo-1])   
            
            if not lon_w_list:
                lon_w_list.append(0)
            if not lon_e_list:
                lon_e_list.append(360)

            if lon_e_list[-1]-lon_w_list[-1] > 0:
                lon_wide_list.append(lon_e_list[-1]-lon_w_list[-1])
            else:
                lon_wide_list.append(360+(lon_e_list[-1]-lon_w_list[-1]))


            ###------ Get the width from north to south -------###
            for la in np.arange(nlat_hemi):
                if (not np.any(LWA_d[la,:])) and (np.any(LWA_d[la-1,:])):
                    lat_n_list.append( lat_hemi[la])
                if (np.any(LWA_d[la,:])) and (not np.any(LWA_d[la-1,:])):
                    lat_s_list.append( lat_hemi[la-1])
            lat_wide_list.append( lat_n_list[-1]-lat_s_list[-1]   ) 
                
                
            ###------ Get the total area ------###
            area_list.append(np.sum(ds_hemi[labels_new[d]==n+1]))
                        
        lat_d.append(lat_list);   lon_d.append(lon_list); lwa_max.append(lwa_max_list); lwa_total.append(lwa_total_list); lwa_mean.append(lwa_mean_list); lwa_label.append(lwa_label_list)
        lon_w.append(lon_w_list); lon_e.append(lon_e_list);  area.append(area_list)
        lat_n.append(lat_n_list); lat_s.append(lat_s_list)
        lon_wide.append(lon_wide_list); lat_wide.append(lat_wide_list)

        
    
    #%%      
    ###------ Pairing wave events across consecutive days ------###

    next_index = []
    distance_thresh= 1e6  ## 1000 km

    for d in np.arange(nday-1):
        next_index_day = np.full(len(lon_d[d]), np.nan)
        
        ### create a matrix that contains the distance between each WE during the consective two days ###   
        shift_L = np.zeros((len(lon_d[d]),len(lon_d[d+1]) ))
        for i in np.arange(len(lon_d[d])):
            for j in np.arange(len(lon_d[d+1])):
                shift_L[i,j] = haversine(lon_d[d][i], lat_d[d][i], lon_d[d+1][j], lat_d[d+1][j])    
        
        ### pair the events from the shortest distance among them ###
        for dd in np.arange( min(len(lon_d[d]), len(lon_d[d+1])) ):
            WE_i, WE_j = np.unravel_index(np.argmin(shift_L), shift_L.shape)
            
            ### note the distance between two paris is also limited by a thrshold ###
            distance_shift = haversine(lon_d[d][WE_i], lat_d[d][WE_i], lon_d[d+1][WE_j], lat_d[d+1][WE_j])

            if distance_shift<distance_thresh:
                next_index_day[WE_i] = WE_j  ### these two events can be paired! ###
                shift_L[WE_i, :] = np.inf    ### make the distance related to these two events to infinity so that we can search the next shortest distance and avoid this pair ###
                shift_L[:, WE_j] = np.inf
            else:
                shift_L[WE_i, :] = np.inf
                shift_L[:, WE_j] = np.inf
                
        next_index.append(next_index_day)
                
    #%%
    ########### Now we begin to track wave events #########

    WE_lat = []; WE_lon = []; WE_date = []; WE_date_index = []; WE_lwa_max = []; WE_lwa_total = []; WE_lwa_mean = []; WE_lwa_label = []
    WE_lon_wide = []; WE_lat_wide = []; WE_area = []; WE_label = []; WE_label_sum = []
    WE_month = []; WE_year = [];  WE_duration = []
    
    WE_peaking_lat = [];  WE_peaking_lon = []
    WE_peaking_date = []; WE_peaking_date_index=[]; WE_peaking_month = []; WE_peaking_year = []
    WE_peaking_lwa_max = []; WE_peaking_lwa_total = []; WE_peaking_lwa_mean = []; WE_peaking_lwa_label = []
    WE_peaking_label = []; WE_peaking_lon_wide = []; WE_peaking_lat_wide = []; WE_peaking_area = []
        
    for d in np.arange(nday-1):
        ### If all wave events within this day are tracked, then go to the next day ###
        if np.all(np.isnan(lon_d[d])):
            continue  
             
        for i in np.arange(len(lon_d[d])):
            ### If this wave evnet is tracked, then go to the next wave event ###
            if np.isnan(lon_d[d][i]):
                continue
            if abs(lat_d[d][i])<30:
                continue
            
            ### Tracking starts ###
            day = 0
            track_lon = []; track_lat = []; track_lon_index = []; track_lat_index = []; 
            track_date = []; track_date_index = []; track_month = []; track_year = []
            track_lwa_max = []; track_lwa_total = []; track_lwa_mean = []; track_lwa_label = []
            track_lon_wide = []; track_area = []; track_lat_wide = []; track_label = []
            
            WE_count = np.zeros((nlat_hemi, nlon))
            WE = np.zeros((nlat_hemi, nlon))
            WE_count2 = []
            
            WE_count[labels_new[d+day]==i+1]+=1
            WE[labels_new[d+day]==i+1]=1            
            WE_count2.append(WE)
            
            track_lon.append(lon_d[d+day][i]); track_lon_index.append(i)               
            track_lat.append(lat_d[d+day][i]); track_lat_index.append(i)            
            track_date.append(Date_year[d+day]); track_date_index.append(d+day); track_month.append(Date_year[d+day][4:6]); track_year.append(Date_year[d+day][0:4])
            track_lwa_max.append(lwa_max[d+day][i]); track_lwa_total.append(lwa_total[d+day][i]); track_lwa_mean.append(lwa_mean[d+day][i]); track_lwa_label.append(lwa_label[d+day][i])
            track_lon_wide.append(lon_wide[d+day][i]); track_lat_wide.append(lat_wide[d+day][i])
            track_area.append(area[d+day][i])
            track_label.append(labels_new[d+day]==i+1)


            
            next_index_pair= next_index[d+day][i] ### find the pair event at next day, it could be nan ###
            if ~np.isnan(next_index_pair):
                next_index_pair = int(next_index_pair)
                
            while ~np.isnan(next_index_pair) and abs(lat_d[d+day+1][next_index_pair])>30:  ### if the next pair is not nan, then keep tracking ###
                track_date.append(Date_year[d+day+1]); track_date_index.append(d+day+1); track_month.append(Date_year[d+day+1][4:6]); track_year.append(Date_year[d+day+1][0:4])
                WE_count[labels_new[d+day+1]==next_index_pair+1]+=1
                WE = np.zeros((nlat_hemi, nlon))
                WE[labels_new[d+day+1]==next_index_pair+1]=1            
                WE_count2.append(WE)
                    
                track_lon.append(lon_d[d+day+1][next_index_pair])
                track_lon_index.append(next_index_pair)
                track_lat.append(lat_d[d+day+1][next_index_pair])
                track_lat_index.append(next_index_pair)
                track_lon_wide.append(lon_wide[d+day+1][next_index_pair])
                track_lat_wide.append(lat_wide[d+day+1][next_index_pair])
                track_area.append(area[d+day+1][next_index_pair])
                track_lwa_max.append(lwa_max[d+day+1][next_index_pair]); track_lwa_total.append(lwa_total[d+day+1][next_index_pair]); track_lwa_mean.append(lwa_mean[d+day+1][next_index_pair]); track_lwa_label.append(lwa_label[d+day+1][next_index_pair])
                track_label.append(labels_new[d+day+1]==next_index_pair+1)

                
                ### interate the day ###
                day+=1
                
                ### if this the last day, then jump out ###
                if d+day+1>nday-1:
                    break
                
                ### if not, then find a next pair ###
                next_index_pair= next_index[d+day][next_index_pair]
                if ~np.isnan(next_index_pair):
                    next_index_pair = int(next_index_pair)
                    
                ### The track is ended until there are no paired events in the next day##
                
            ### when there are no paired events in the next day, the track is end ###
            WE_lon.append(track_lon)
            WE_lat.append(track_lat)
            WE_date.append(track_date)
            WE_date_index.append(track_date_index)
            WE_month.append(track_month)
            WE_year.append(track_year)
            WE_lwa_max.append(track_lwa_max)
            WE_lwa_total.append(track_lwa_total)
            WE_lwa_mean.append(track_lwa_mean)
            WE_lwa_label.append(track_lwa_label)
            WE_lon_wide.append(track_lon_wide)
            WE_lat_wide.append(track_lat_wide)
            WE_area.append(track_area)
            WE_label_sum.append(WE_count)
            WE_label.append(track_label)
                
            # Find the peaking date information of this wave event #
            lwa_peak = max(track_lwa_max)
            lwa_peak_index = track_lwa_max.index(lwa_peak)
            
            WE_peaking_lwa_max.append(lwa_peak); WE_peaking_lwa_total.append(track_lwa_total[lwa_peak_index]); WE_peaking_lwa_mean.append(track_lwa_mean[lwa_peak_index]); WE_peaking_lwa_label.append(track_lwa_label[lwa_peak_index])
            WE_peaking_lon.append(track_lon[lwa_peak_index]); WE_peaking_lat.append(track_lat[lwa_peak_index])
            WE_peaking_date.append(track_date[lwa_peak_index]); WE_peaking_date_index.append(track_date_index[lwa_peak_index])
            WE_peaking_month.append(track_month[lwa_peak_index]); WE_peaking_year.append(track_year[lwa_peak_index])
            WE_peaking_lon_wide.append(track_lon_wide[lwa_peak_index]); WE_peaking_lat_wide.append(track_lat_wide[lwa_peak_index])
            WE_peaking_area.append(track_area[lwa_peak_index])
            WE_peaking_label.append(track_label[lwa_peak_index])
            WE_duration.append(day+1)
         

            ### Once we successfully detected a wave event, we mark that with nan to avoid double counting ###
            for dd in np.arange(day+1):
                lon_d[d+dd][track_lon_index[dd]] = np.nan
                lat_d[d+dd][track_lat_index[dd]] = np.nan
      
                   
          
    #%%
    ###------ After tracking all the wave events, now we do some filtering ------### 

    Extreme_thresh = np.percentile(np.array(WE_peaking_lwa_max),99) 
    Extreme_thresh1 = np.percentile(np.array(WE_peaking_lwa_max),45) 
    Extreme_thresh2 = np.percentile(np.array(WE_peaking_lwa_max),55)


    Extreme_peaking_date = [] ; Extreme_peaking_lon = []; Extreme_peaking_lat = [];Extreme_peaking_lwa_max = []; Extreme_peaking_lwa_total = []; Extreme_peaking_lwa_mean = []
    Extreme_peaking_lon = []; Extreme_peaking_lat = []; Extreme_peaking_lon_wide = []; Extreme_peaking_lat_wide = []; Extreme_peaking_area = []
    Extreme_peaking_month = []; Extreme_peaking_year = []; Extreme_peaking_label = []; Extreme_peaking_lwa_label = []

    Extreme_date = []; Extreme_lon = []; Extreme_lat = [];Extreme_lwa_max = []; Extreme_lwa_total = []; Extreme_lwa_mean = []
    Extreme_duration =[]; Extreme_lon_wide = []; Extreme_lat_wide = []; Extreme_area = []; Extreme_label = [];  Extreme_lwa_label = []
    Extreme_month = []; Extreme_year = []
    
    Extreme_freq = np.zeros((nlat_hemi, nlon))         ### The final ExtremeWE frequency/total number
    for n in np.arange(len(WE_date)):
        # if WE_peaking_lon_wide[n]>15 and WE_duration[n]>=4 and WE_peaking_lwa_max[n]>Extreme_thresh:  
        # if WE_peaking_lon_wide[n]>15 and WE_duration[n]>=4 and WE_peaking_lwa_max[n]>Extreme_thresh1 and WE_peaking_lwa_max[n]<Extreme_thresh2: 
        if WE_peaking_lon_wide[n]>15 and WE_duration[n]>=4:
            Extreme_peaking_lwa_max.append(WE_peaking_lwa_max[n])
            Extreme_peaking_lwa_total.append(WE_peaking_lwa_total[n])
            Extreme_peaking_lwa_mean.append(WE_peaking_lwa_mean[n])
            Extreme_peaking_lon.append(WE_peaking_lon[n])
            Extreme_peaking_lat.append(WE_peaking_lat[n])
            Extreme_peaking_date.append(WE_peaking_date[n])
            Extreme_peaking_month.append(WE_peaking_month[n])
            Extreme_peaking_year.append(WE_peaking_year[n])
            Extreme_peaking_lon_wide.append(WE_peaking_lon_wide[n])
            Extreme_peaking_lat_wide.append(WE_peaking_lat_wide[n])
            Extreme_peaking_area.append(WE_peaking_area[n])
            Extreme_peaking_label.append(WE_peaking_label[n])
            Extreme_peaking_lwa_label.append(WE_peaking_lwa_label[n])

            Extreme_date.append(WE_date[n])
            Extreme_month.append(WE_month[n])
            Extreme_year.append(WE_year[n])
            Extreme_lwa_max.append(WE_lwa_max[n])
            Extreme_lwa_total.append(WE_lwa_total[n])
            Extreme_lwa_mean.append(WE_lwa_mean[n])
            Extreme_lon.append(WE_lon[n])
            Extreme_lat.append(WE_lat[n]) 
            Extreme_lon_wide.append(WE_lon_wide[n])
            Extreme_lat_wide.append(WE_lat_wide[n])
            Extreme_area.append(WE_area[n]) 
            Extreme_duration.append(WE_duration[n])
            Extreme_label.append(WE_label[n])
            Extreme_lwa_label.append(WE_lwa_label[n])
            
            Extreme_freq += WE_label_sum[n]  ### add the frequency of this event to the total frequency
            

    #%%
    
    ###------ Save the results ------###
    output_path = "/scratch/bell/liu3315/ERA5/LWA_theta/WE_theta/"+season_id+"/WE_"+hemisphere_id+"_"+isentrope_name+"_"+variable_id1+"/" + str(nyear)+"/"
    
    with open(output_path+"WE_date", "wb") as fp:
        pickle.dump(Extreme_date, fp)

    with open(output_path+"WE_peaking_date", "wb") as fp:
        pickle.dump(Extreme_peaking_date, fp)
        
    with open(output_path+"WE_lon", "wb") as fp:
        pickle.dump(Extreme_lon, fp)

    with open(output_path+"WE_peaking_lon", "wb") as fp:
        pickle.dump(Extreme_peaking_lon, fp)
                    
    with open(output_path+"WE_lat", "wb") as fp:
        pickle.dump(Extreme_lat, fp)

    with open(output_path+"WE_peaking_lat", "wb") as fp:
        pickle.dump(Extreme_peaking_lat, fp)
                    
    with open(output_path+"WE_LWA_max", "wb") as fp:
        pickle.dump(Extreme_lwa_max, fp)

    with open(output_path+"WE_peaking_LWA_max", "wb") as fp:
        pickle.dump(Extreme_peaking_lwa_max, fp)
        
    with open(output_path+"WE_LWA_total", "wb") as fp:
        pickle.dump(Extreme_lwa_total, fp)

    with open(output_path+"WE_peaking_LWA_total", "wb") as fp:
        pickle.dump(Extreme_peaking_lwa_total, fp)
        
    with open(output_path+"WE_LWA_mean", "wb") as fp:
        pickle.dump(Extreme_lwa_mean, fp)

    with open(output_path+"WE_peaking_LWA_mean", "wb") as fp:
        pickle.dump(Extreme_peaking_lwa_mean, fp)

    with open(output_path+"WE_label", "wb") as fp:
        pickle.dump(Extreme_label, fp)

    with open(output_path+"WE_duration", "wb") as fp:
        pickle.dump(Extreme_duration, fp)
        
    with open(output_path+"WE_lon_wide", "wb") as fp:
        pickle.dump(Extreme_lon_wide, fp)

    with open(output_path+"WE_peaking_lon_wide", "wb") as fp:
        pickle.dump(Extreme_peaking_lon_wide, fp)
    
    with open(output_path+"WE_lat_wide", "wb") as fp:
        pickle.dump(Extreme_lat_wide, fp)
        
    with open(output_path+"WE_peaking_lat_wide", "wb") as fp:
        pickle.dump(Extreme_peaking_lat_wide, fp)

    with open(output_path+"WE_area", "wb") as fp:
        pickle.dump(Extreme_area, fp)
        
    with open(output_path+"WE_peaking_area", "wb") as fp:
        pickle.dump(Extreme_peaking_area, fp)

    with open(output_path+"WE_peaking_label", "wb") as fp:
        pickle.dump(Extreme_peaking_label, fp)

    with open(output_path+"WE_lwa_label", "wb") as fp:
        pickle.dump(Extreme_lwa_label, fp)
    

    np.save(output_path+"WE_freq.npy", Extreme_freq)



#%%
if __name__ == "__main__":
    
    ###------ File paths for all dates ------###
    path = glob.glob(r"/scratch/bell/liu3315/ERA5/LWA_theta/"+"*.nc")
    path.sort()
    Date = []
    for i in np.arange(len(path)):
        Date.append(path[i][-13:-3])

    ###------ Parameters ------###
    variable_id1 = 'RWP'
    variable_id2 = 'RWP'
    season_id = 'NDJFM'
    hemisphere_id = 'NH'
    isentrope_id = 1  ## 0:315K; 1:320K; 2:325K; 3:330K, 4:335K,5:340K,6:345K
    isentrope_name = '320K'

    ###------ Parallel processing over years ------###
    numlist = np.arange(1979,2023)
    numlist= numlist.astype(int)
    numlist = numlist.tolist()
    n_core=64
    pool = multiprocessing.Pool(processes=n_core)
    pool.map(WE_track, numlist)
    pool.close()
    pool.join()


