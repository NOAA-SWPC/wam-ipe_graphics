"""
Filled contours
---------------

An example of contourf on manufactured data.

"""
from __future__ import print_function

#import matplotlib
#matplotlib.use('Agg')
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from netCDF4 import Dataset
import glob
import cartopy
import cartopy.crs as ccrs
from datetime import datetime,timedelta
from multiprocessing import Pool
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def plot(file):
    # get these from argument parameters
    nticks = 7
    title_size = 10
    print(file)
    proj = ccrs.PlateCarree(central_longitude=0)

    nc_fid = Dataset("{}".format(file), "r", format="NETCDF4")
    lon = nc_fid.variables['lon'] # longitude
    lat = nc_fid.variables['lat'] # latitude
    tec = nc_fid.variables['tec'] # TEC
    nmf2 = nc_fid.variables['NmF2'] # nmf2
    hmf2 = nc_fid.variables['HmF2'] # nmf2

    try: current_date = datetime.strptime(nc_fid.variables['fcst_date'],"%Y%m%d_%H%M%S")
    except : current_date = datetime.now()
    timestamp = current_date.strftime("%Y%m%d_%H%M%S")
    try: run_date = datetime.strptime(nc_fid.variables['init_date'],"%Y%m%d_%H%M%S")
    except : run_date = datetime.now()

    latvals = lat[:]

    lonvals = lon[:]

    tecvals = tec[:]

    nmf2vals = nmf2[:]
    hmf2vals = hmf2[:]
    print("NmF2 max: {:.2e}".format(nmf2vals.max()))
    print("HmF2 max: {:.1f}".format(hmf2vals.max()))

    cmap = plt.get_cmap('Blues',256)
    cmap2 = plt.get_cmap('Purples',256)
    cmap3 = plt.get_cmap('gist_ncar',256)


    fig, axes = plt.subplots(1, 3, subplot_kw=dict(projection=proj), figsize=(16,4))
    fig.subplots_adjust(hspace=0.01,wspace=0.01,top=1.3,left=0.1)

    ax  = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]

    contour_plot = ax.contourf(lonvals, latvals, tecvals, np.linspace(0,50,100), extend='both', transform=ccrs.PlateCarree(central_longitude=0), cmap=cmap)

#    axpos = ax.get_position()
#    pos_x = axpos.x0+axpos.width + 0.01# + 0.25*axpos.width
#    pos_y = axpos.y0
#    cax_width = 0.04
#    cax_height = axpos.height
    #create new axes where the colorbar should go.
    #it should be next to the original axes and have the same height!
#    pos_cax = fig.add_axes([pos_x,pos_y,cax_width,cax_height])

#    plt.colorbar(contour_plot, cax=pos_cax, ticks=np.linspace(0,100,7))
    cax, kw = mpl.colorbar.make_axes(ax,cmap=cmap,location='bottom',pad=0.05,shrink=0.7)
    out=fig.colorbar(contour_plot,cax=cax,ticks=np.linspace(0,50,nticks),**kw)
    #cb = mpl.colorbar.ColorbarBase(ax,cmap=cmap,orientation='vertical')
    #cb.set_label('TEC (TECu)')  

    ax.coastlines(alpha=0.2)
    ax.gridlines(alpha=0.1)

    # ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black', color='black')
    ax.set_global()
    tec_title = ax.text(0.5,1.05,'TEC',fontsize=title_size,transform=ax.transAxes,horizontalalignment='center')

# Now the nmf2 plot....
    contour_plot = ax2.contourf(lonvals, latvals, nmf2vals, np.linspace(0,2.e12,100), extend='max',transform=ccrs.PlateCarree(),cmap=cmap2)
    nmf2_title = ax2.text(0.5,1.05,'NmF2',fontsize=title_size,transform=ax2.transAxes,horizontalalignment='center')
    cax2, kw = mpl.colorbar.make_axes(ax2,cmap=cmap,location='bottom',pad=0.05,shrink=0.7)
    out=fig.colorbar(contour_plot,cax=cax2,ticks=np.linspace(0,2.e12,nticks),**kw)

    ax2.coastlines(alpha=0.2)
    ax2.gridlines(alpha=0.1)
    ax2.set_global()

# Now the hmf2 plot....
    contour_plot = ax3.contourf(lonvals, latvals, hmf2vals, np.linspace(0,1200,100),extend='max',transform=ccrs.PlateCarree(),cmap=cmap3,vmin=0,vmax=1200)
    cax3, kw = mpl.colorbar.make_axes(ax3,cmap=cmap,location='bottom',pad=0.05,shrink=0.7)
    out=fig.colorbar(contour_plot,cax=cax3,ticks=np.linspace(0,1200,nticks),**kw)
    hmf2_title = ax3.text(0.5,1.05,'HmF2',fontsize=title_size,transform=ax3.transAxes,horizontalalignment='center')
    ax3.coastlines(alpha=0.2)
    ax3.gridlines(alpha=0.1)
    ax3.set_global()
#    fmt = FuncFormatter(lambda x, pos: str(int(x)))
    cax.tick_params(labelsize=6)
#    cax.xaxis.set_major_formatter(fmt)
    cax2.tick_params(labelsize=6)
#    cax2.xaxis.set_major_formatter(fmt)
    cax3.tick_params(labelsize=6)
#    cax3.xaxis.set_major_formatter(fmt)
    plt.suptitle("wfs.{}/{}: {}".format(run_date.strftime("%Y%m%d"), run_date.strftime("%H"), current_date.strftime("%-m/%-d/%Y %H:%MUT")),fontdict={'family':'monospace'})
#    plt.show()

    plt.savefig("trio_{}.png".format(timestamp))
    plt.close()

def main():
    # paralellize!
#    print('main')
    parser = ArgumentParser(description='plot tec, nmf2, hmf2', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--file',   help='filename', type=str, required=True)
    args = parser.parse_args()
    plot(args.file)
#    with Pool(processes=4) as pool:
#        pool.map(plot, range(1018))

if __name__ == '__main__':
    main()
