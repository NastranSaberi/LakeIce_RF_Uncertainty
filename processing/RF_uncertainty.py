#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:28:53 2021

@author: nsaberi
"""
import os
import pickle
import subprocess
import time
from os import listdir
from os.path import isfile, join
from subprocess import call

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import sklearn
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from osgeo import gdal
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import Uncertainty as unc

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


s_time = time.time()

dir_main='/home/nsaberi/projects/def-ka3scott/nsaberi/RF_uncertainty/'
path_images=dir_main+'images/'


csv_dir=dir_main+'codes/MOD02_LS.csv'
df = pd.read_csv(csv_dir)
clean_dataset(df)
x=df[['B01', 'B02', 'B03', 'B04', 'B06', 'B07']] #, 'SZA'
y=df[['Label']]
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3) # 70% training and 30% test
clf = joblib.load(dir_main+"random_forest.joblib")
print('RF is loaded ...')
files = [f for f in listdir(path_images) if isfile(join(path_images, f))]
inshape = dir_main+'images/shp/LakeErie.shp'
for kk in range(1,len(files)):
    inraster = path_images+files[kk]
    temp=files[kk]
    print('processing'+files[kk])
    outraster=dir_main+'temp_masked_images/'+temp[0:len(temp)-4]+'_masked.tif'
    subprocess.call(['gdalwarp', inraster, outraster, '-cutline', inshape,
                 '-crop_to_cutline'])
    dataset = gdal.Open(outraster, gdal.GA_ReadOnly) 
    band = dataset.GetRasterBand(1)
    arr = band.ReadAsArray()
    x_num=dataset.RasterXSize
    y_num=dataset.RasterYSize
    del arr
    arr=np.zeros((y_num,x_num,6))
    data=np.zeros((y_num*x_num,6))
    band= dataset.GetRasterBand(1)   
    arr_masked = band.ReadAsArray()
    landmask_arr=arr_masked
    landmask_arr[arr_masked==0]=np.nan
    
    figure=plt.imshow(arr_masked)#, vmin=0., vmax=1
    plt.savefig(dir_main+'temp_masked_images/'+temp[0:len(temp)-4]+'_masked.jpg',dpi=1000)
### Reshape rasters to one-D array to look like X_tes
    i=0
    for ii in [0,1,2,3,5,6]:
        dataset = gdal.Open(outraster, gdal.GA_ReadOnly) 
        band= dataset.GetRasterBand(ii+1)   
        arr[:,:,i] = band.ReadAsArray()
        temp=arr[:,:,i].reshape(arr[:,:,1].shape[0]*arr[:,:,1].shape[1],1)
        data[:,i]=temp[:,0]
        i=i+1
    
    l=0
    y_pred=clf.predict(data)
    y_pred_map=np.reshape(y_pred, (y_num,x_num))
    plt.figure(l)
    plt.gcf().clear()
    cmap1 = plt.get_cmap('viridis') #bwr
    newcmp = ListedColormap(cmap1(np.linspace(0, 1, 3)))#
    newcmp.set_under('white') # Color for values less than vmin
    XX=y_pred_map
    XX[np.isnan(landmask_arr)]=-1
    figure0 = plt.imshow(XX, cmap=newcmp,vmin=0, vmax=3)
    values = np.unique(y_pred_map.ravel())
    colors = [ figure0.cmap(figure0.norm(value)) for value in values]
    cbar = plt.colorbar(figure0,ticks=[0.5,1.5,2.5])
    cbar.ax.set_yticklabels(['water=1', 'ice=2','cloud=3'])  
    plt.show()
    plt.savefig(dir_main+'results/RF_map_'+inraster[len(inraster)-27:len(inraster)-13]+'.jpg',dpi=1000)
    l=l+1

#######     UNCERTAINTY     ############
    print('Uncertainty calculation')
    data[np.isnan(data)]=-1 ## when the lake is not fully covered and we need mosaicing
    total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.model_uncertainty(clf, data, X_train, y_train)
    print('uncertainty is calculated; start plotting'+files[kk])
# Aleatoric uncertianty for data
    aleatoric_uncertainty_map = np.reshape(aleatoric_uncertainty, (y_num,x_num))
    aleatoric_uncertainty_map = np.reshape(aleatoric_uncertainty, (y_num,x_num))
    epistemic_uncertainty_map= np.reshape(epistemic_uncertainty, (y_num,x_num))
    total_uncertainty_map = np.reshape(total_uncertainty, (y_num,x_num))

    # filename = dir_main+'results/uncertainty.pkl'
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    # with open(filename, 'wb') as f:    
    #     pickle.dump(total_uncertainty, f)
    #     pickle.dump(epistemic_uncertainty, f)
    #     pickle.dump(aleatoric_uncertainty, f)

#######     More Plots      ##############
    min_mapping=0 
    max_mapping=1.44435

    plt.figure(l)
    plt.gcf().clear()
    cmap1 = plt.get_cmap('bwr') #bwr
    newcmp = ListedColormap(cmap1(np.linspace(min_mapping, max_mapping, 256)))#
    newcmp.set_under('white') # Color for values less than vmin
    XX=aleatoric_uncertainty_map
    XX[np.isnan(landmask_arr)]=-1
    figure1=plt.imshow(XX,cmap=newcmp,vmin=min_mapping,vmax=max_mapping)#, vmin=0., vmax=1
    cbar = plt.colorbar(figure1,extend='min')
    cbar.set_label(' Aleatoric uncertainty', rotation=270,labelpad=15)
    plt.savefig(dir_main+'results/ale'+inraster[len(inraster)-27:len(inraster)-13]+'.jpg',dpi=1000)
    l=l+1

###
    plt.figure(l)
    plt.gcf().clear()
    cmap1 = plt.get_cmap('bwr') #bwr
    newcmp = ListedColormap(cmap1(np.linspace(min_mapping, max_mapping, 256)))#
    newcmp.set_under('white') # Color for values less than vmin
    XX=epistemic_uncertainty_map
    XX[np.isnan(landmask_arr)]=-1
    figure1=plt.imshow(XX,cmap=newcmp,vmin=min_mapping,vmax=max_mapping)#, vmin=0., vmax=1
    cbar = plt.colorbar(figure1,extend='min')
    cbar.set_label(' Epistemic uncertainty', rotation=270,labelpad=15)
    plt.savefig(dir_main+'results/eps'+inraster[len(inraster)-27:len(inraster)-13]+'.jpg',dpi=1000)
    l=l+1

###

    plt.figure(l)
    plt.gcf().clear()
    cmap1 = plt.get_cmap('bwr') #bwr
    newcmp = ListedColormap(cmap1(np.linspace(min_mapping, max_mapping, 256)))#
    newcmp.set_under('white') # Color for values less than vmin
    XX=total_uncertainty_map
    XX[np.isnan(landmask_arr)]=-1
    figure1=plt.imshow(XX,cmap=newcmp,vmin=min_mapping,vmax=max_mapping)#, vmin=0., vmax=1
    cbar = plt.colorbar(figure1,extend='min')
    cbar.set_label(' Total uncertainty', rotation=270,labelpad=15)
    plt.savefig(dir_main+'results/total'+inraster[len(inraster)-27:len(inraster)-13]+'.jpg',dpi=1000)

    del landmask_arr
######



e_time = time.time()
run_time = int(e_time - s_time)
print(f"time :{run_time/3600}h")
