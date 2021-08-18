import tkinter as tk
from tkcalendar import DateEntry
import datetime
import pandas as pd
import time
import numpy as np
import os
from tqdm import tqdm
import ee
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import scipy
from scipy import stats
from scipy.stats import pearsonr
import time
import cartopy.crs as ccrs
import geopandas
import cartopy.io.img_tiles as cimgt
os.environ['PROJ_LIB'] = 'C:/Users/owner/Anaconda3/Lib/site-packages/mpl_toolkits/basemap'
from mpl_toolkits.basemap import Basemap
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
import matplotlib.ticker as mticker

import warnings
warnings.filterwarnings("ignore")


root = tk.Tk()
root .title('Remote Sensing Rainfall Retriever with GEE 🛰️🌧️')
try:
    root.iconbitmap(r"C:\Users\owner\Google Drive 2\Python Library\GEE\gee2.ico")
except:
    pass

stamen_terrain = cimgt.Stamen('terrain-background')
ll_proj = ccrs.PlateCarree()

ee.Initialize()

def table_on_plot(df):
    df = df.round(3)
    stats = df.mean(1).mean(), df.median(1).median(), df.min(1).min(), df.max(1).max(), df.std(1).std()
    stats = np.round(stats, 3)
    stats = pd.DataFrame(data = stats, index = ['Mean', 'Median', 'Min', 'Max', 'Std'], columns = ['Statistics'])
    table = plt.table(cellText=stats.values, colWidths = [0.5] * len(stats.columns),
          rowLabels = stats.index, cellLoc = 'left', rowLoc = 'left', loc = 'upper right', edges='open') # Adjust table size loc and allignment
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(.5, 1.5)

def plot_station(station, nice_map = True):
    if nice_map:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ll_proj)
        m=Basemap()
        m.etopo(scale=0.5, alpha=0.5)
        m.shadedrelief()
        m.drawcoastlines(0.75)
        parallels = np.arange(-90.,91.,15.)
        m.drawparallels(parallels,labels=[False,True,True,False])
        meridians = np.arange(-180.,181.,25.)
        m.drawmeridians(meridians,labels=[True,False,False,True])
        ax.scatter(station[0], station[1], marker = '^', color = 'red', s = 100, zorder = 100)
        ax.set_extent([-180, 180, -90, 90])
        plt.show();
    else:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ll_proj)
        ax.coastlines()
        ax.scatter(station[0], station[1], marker = '^', color = 'red', s = 100, zorder = 100)
        ax.set_extent([-180, 180, -90, 90])
        plt.show();

def GEE(date_start, date_stop, image_coll, patch, band2): 
    im_coll = ee.ImageCollection(image_coll).filterDate(ee.Date(date_start),ee.Date(date_stop)).select(band2)
    
    acq_times = im_coll.aggregate_array('system:time_start').getInfo()
    avalaible_dates = [time.strftime('%x', time.gmtime(acq_time/1000)) for acq_time in acq_times]
    im_list = im_coll.toList(im_coll.size())
    variable_list = ee.List([])
   
    def reduce_dataset_region(image, list):
        local_image = image.reduceRegion(
            reducer=ee.Reducer.median(),
            geometry=patch,
            scale=1,
            crs = "epsg:4326")

        return ee.List(list).add(local_image)

    reduced_dataset = im_coll.iterate(reduce_dataset_region, variable_list)
    reduced_dataset_dict = reduced_dataset.getInfo()
    df1 = pd.DataFrame(data = reduced_dataset_dict, index = avalaible_dates, columns = [band2]).dropna()

    df1.index = pd.to_datetime(df1.index)
    return df1

def rain_gee(point, date_start, date_stop):

    def show_words():
        label.configure(text=msg)
    
    global msg
    msg = 'GPM...'
    image_coll = "NASA/GPM_L3/IMERG_MONTHLY_V06"
    band2 = 'precipitation'
    label.config(text = msg)
    GPM = GEE(date_start, date_stop, image_coll, point, band2)

    if float(lat.get()) >= -50 and float(lat.get()) <= 50 and float(lon.get()) >= -126 and float(lon.get()) <= 153:
        if var.get() == 1:
            msg = 'CHIRPS...'
            image_coll = 'UCSB-CHG/CHIRPS/PENTAD'
        else:
            msg = 'CHIRPS...'
            image_coll = 'UCSB-CHG/CHIRPS/DAILY'
        band2 = 'precipitation'
        label.config(text = msg)
        label.update_idletasks()
        CHIRPS = GEE(date_start, date_stop, image_coll, point, band2)
    else:
        CHIRPS = pd.DataFrame(index = pd.date_range(start=date_start, end=date_stop, freq='D'))
        CHIRPS['prec'] = np.nan

    msg = 'ECMWF...'
    image_coll = "ECMWF/ERA5/MONTHLY"
    band2 = 'total_precipitation'
    label.config(text = msg)
    label.update_idletasks()
    ECMWF = GEE(date_start, date_stop, image_coll, point, band2)

    if float(lat.get()) >= -50 and float(lat.get()) <= 50:
        msg = 'TRMM...'
        image_coll = "TRMM/3B43V7"
        band2 = 'precipitation'
        label.config(text = msg)
        label.update_idletasks()    
        TRMM = GEE(date_start, date_stop, image_coll, point, band2)
    else:
        TRMM = pd.DataFrame(index = pd.date_range(start=date_start, end=date_stop, freq='D'))
        TRMM['prec'] = np.nan

    if float(lat.get()) >= -60 and float(lat.get()) <= 60:
        msg = 'PERSIANN...'
        image_coll = "NOAA/PERSIANN-CDR"
        band2 = 'precipitation'
        label.config(text = msg)
        label.update_idletasks()
        PERSIANN = GEE(date_start, date_stop, image_coll, point, band2)
    else:
        PERSIANN = pd.DataFrame(index = pd.date_range(start=date_start, end=date_stop, freq='D'))
        PERSIANN['prec'] = np.nan

    if var.get() == 1:
        GPM_r = GPM.resample('M').sum() * 24 * 30
        TRMM_r = TRMM.resample('M').sum() * 24 * 30
        ECMWF_r = ECMWF.resample('M').sum() * 24 * 30
        CHIRPS_r = CHIRPS.resample('M').sum()
        PERSIANN_r = PERSIANN.resample('M').sum()
        comparison = pd.concat([GPM_r, TRMM_r, ECMWF_r, CHIRPS_r, PERSIANN_r], 1)
        comparison.columns = ['GPM', 'TRMM', 'ECMWF', 'CHRIPS', 'PERSIANN']
        
    else:
        CHIRPS_r = CHIRPS.resample('D').sum()
        PERSIANN_r = PERSIANN.resample('D').sum()
        comparison = pd.concat([CHIRPS_r, PERSIANN_r], 1)
        comparison.columns = ['CHRIPS', 'PERSIANN']
    
    # comparison['Median'] = comparison.median(1)
    # label.destroy()
    label.config(text = 'Median Rainfall: {} mm'.format(np.round(comparison.median(1).median(), 2)))
    
    fig, ax = plt.subplots(1, 1, figsize=[8,6])
    plt.ylabel('Precipitation [mm]')
    plt.xlabel('Dates')
    comparison.plot(kind = 'bar', ax = ax)
    comparison['Date'] = comparison.index
    plt.legend(loc='upper left')
    ticklabels = ['']*len(comparison)
    skip = len(comparison)//12
    ticklabels[::skip] = comparison['Date'].iloc[::skip].dt.strftime('%Y-%m-%d')
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(ticklabels))
    fig.autofmt_xdate()
    table_on_plot(comparison)
    plt.show();

    if CheckVar1.get():
        try:
            currdir = os.getcwd()
            tempdir = filedialog.askdirectory(parent=root, initialdir=currdir, title='Please select a directory')
            comparison.to_excel(r'{}/Rainfall.xlsx'.format(tempdir), engine='xlsxwriter')
        except:
            pass
    return comparison

def run():
    global comparison
    start_time = time.time()
    d1 = datetime.datetime.strptime(str(cal1.get_date()), '%Y-%m-%d')
    d2 = datetime.datetime.strptime(str(cal2.get_date()), '%Y-%m-%d')
    x, y = float(lat.get()), float(lon.get())
    comparison = rain_gee(ee.Geometry.Point(list([y, x])), d1, d2)
    time_ = (time.time() - start_time)
    blank.config(text = "--- %.2fs seconds ---" % (time.time() - start_time))
    return comparison

def run2():
    x, y = float(lat.get()), float(lon.get())
    plot_station([y, x], nice_map = False)

cal1 = DateEntry(root, width=12, year=2008, month=1, day=1, 
background='darkblue', foreground='white', borderwidth=2, date_pattern='dd/mm/y')
cal1.grid(column = 1, row=0, sticky='w')

l1 = tk.Label(root, text = "Starting Day", width = 15)
l1.grid(column = 0, row=0, sticky='w')

cal2 = DateEntry(root, width=12, year=2008, month=12, day=31, 
background='darkblue', foreground='white', borderwidth=2, date_pattern='dd/mm/y')
cal2.grid(column = 1, row=1, sticky='w')

l2 = tk.Label(root, text = "Ending Day", width = 15)
l2.grid(column = 0, row=1, sticky='w')

lat = tk.Entry(root, width = 15)
lat.grid(column = 1, row=2, sticky='w')
l3 = tk.Label(root, text = "Latitude", width = 15)
l3.grid(column = 0, row=2, sticky='w')
lat.insert(0, 40.538)

lon = tk.Entry(root, width = 15)
lon.grid(column = 1, row=3, sticky='w')
l4 = tk.Label(root, text = "Longitude", width = 15)
l4.grid(column = 0, row=3, sticky='w')
lon.insert(0, 22.996)

l5 = tk.Label(root, text = "Powered by Dimos Touloumidis")
l5.grid(column = 0, row=12, sticky='w')
l5.config(font=("Courier", 8))

blank =tk.Label(root)
blank.grid(column = 4, row=1, sticky='w')
blank_l = tk.Label(root, text = "Elapsed Time", width = 15)
blank_l.grid(column = 4, row=0, sticky='w')

CheckVar1 = tk.IntVar()

C1 = tk.Checkbutton(root, text = "Excel", variable = CheckVar1, \
                 onvalue = 1, offvalue = 0, height=1, \
                 width = 15)

C1.grid(row=2, column=2, sticky='w')

label1 = tk.Label(root, text = "Time Scale?")
label1.grid(row=0, column = 2, columnspan=2)

var = tk.IntVar(value = 2)
Rb1 = tk.Radiobutton(text="Monthly", variable=var, value=1, width = 15)
Rb1.grid(row=1, column=2, sticky='w')

Rb2 = tk.Radiobutton(text="Daily", variable=var, value=2, width = 15)
Rb2.grid(row=1, column=3, sticky='w')

b1 = tk.Button(root, text = "Get Rainfall!", command = run, width = 15)
b1.grid(column = 3, row=3, sticky='w')

b2 = tk.Button(root, text = "Plot station", command = run2, width = 15)
b2.grid(column = 3, row=4, sticky='w')

b3 = tk.Button (root, text='Exit Application', command=root.destroy, width = 15)
b3.grid(column = 4, row=4, sticky='w')

label = tk.Label(root, width = 25)
label.grid(column = 4, row=3, sticky='w')

root.mainloop()