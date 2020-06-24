"""
Filled contours
---------------

An example of contourf on manufactured data.

"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from netCDF4 import Dataset

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature



def ncdump(nc_fid, verb=True):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset. 

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    '''
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print( "\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print( '\t\t%s:' % ncattr,\
                      repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print( "\t\tWARNING: %s does not contain variable attributes" % key)

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print( "NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print( '\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print( "NetCDF dimension information:")
        for dim in nc_dims:
            print( "\tName:", dim )
            print( "\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print( "NetCDF variable information:")
        for var in nc_vars:
            if var not in nc_dims:
                print( '\tName:', var)
                print( "\t\tdimensions:", nc_fid.variables[var].dimensions)
                print( "\t\tsize:", nc_fid.variables[var].size)
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars


def main():
	nc_fid = Dataset("/Users/george/Downloads/ipe.20130316_010000_nmf2.nc", "r", format="NETCDF4")
	# nc_fid = Dataset("/Users/george/Downloads/IPE_Ne.geo.202006041900.nc4", "r", format="NETCDF4")
	nc_attrs, nc_dims, nc_vars = ncdump(nc_fid)
	lon = nc_fid.variables['lon'] # longitude
	lat = nc_fid.variables['lat'] # latitude
	# lon = nc_fid.variables['longitude'] # longitude
	# lat = nc_fid.variables['latitude'] # latitude
	# alt = nc_fid.variables['altitude'] # latitude
	tec = nc_fid.variables['tec'] # TEC
	nmf2 = nc_fid.variables['nmf2'] # nmf2
	hmf2 = nc_fid.variables['hmf2'] # nmf2
	latvals = lat[:]
	lonvals = lon[:]
	tecvals = tec[:]
	nmf2vals = nmf2[:] / 1.e12
	hmf2vals = hmf2[:]
	print("NmF2 max: {:.2e}".format(nmf2vals.max()))
	print("HmF2 max: {:.1f}".format(hmf2vals.max()))

	cmap = plt.get_cmap('Blues',256)
	cmap2 = plt.get_cmap('Reds',256)
	cmap3 = plt.get_cmap('Greens',256)
	# cmap3 = plt.get_cmap('gist_ncar',256)
	# cmap = plt.get_cmap('cividis',256)
	# cmap2 = plt.get_cmap('viridis',256)
	# cmap3 = plt.get_cmap('jet',256)

	# text_color = (0.9,0.9,0.9)
	text_color = (0.,0.,0.)
	edge_color = (0.0,0.4,0.0)
	# background_color = (0.1,0.1,0.1)
	background_color = (0.9,0.9,0.9)

	fig=plt.figure(figsize=(16,9),facecolor=background_color)

	main_title_size = 20
	main_title = fig.text(0.02,0.89,'2020-06-04 12:15 UTC',fontsize=48,horizontalalignment='left',color=text_color)
	main_title = fig.text(0.02,0.82,'Global Ionosphere',fontsize=32,horizontalalignment='left',color=text_color)
	main_title = fig.text(0.04,0.76,'Model: WAM-IPE',fontsize=20,horizontalalignment='left',color=text_color)
	main_title = fig.text(0.02,0.025,'Space Weather Prediction Center',fontsize=16,horizontalalignment='left',color=text_color)

	projection = ccrs.PlateCarree(central_longitude=0)
	the_plots = fig.subplots(2, 2, subplot_kw=dict(projection=projection))
	fig.subplots_adjust(left=0.03, right=0.98, bottom=0.02, top=0.98, wspace=0.01, hspace=0.0)
	nticks = 5
	nticks_tec = 6
	title_size = 14
	n_contours = 20

	ax0 = the_plots[0,0]
	ax0.set_visible(False)

	max_tec = 100.
	min_tec = 0.0
	max_nmf2 = 5.
	min_nmf2 = 0.0
	landcolor = (1,1,1)

	ax1 = the_plots[0,1]
	contour_plot = ax1.contourf(lonvals, latvals, tecvals, np.linspace(min_tec,max_tec,n_contours), extend='max', transform=ccrs.PlateCarree(central_longitude=0), cmap=cmap)
	tec_title = ax1.text(0.5,1.05,'Total Electron Content (TEC)',fontsize=title_size,transform=ax1.transAxes,horizontalalignment='center',color=text_color)

	cax, kw = mpl.colorbar.make_axes(ax1,cmap=cmap,pad=0.03,shrink=0.6)
	cb1=fig.colorbar(contour_plot,cax=cax,ticks=np.linspace(min_tec,max_tec,nticks_tec),**kw)
	cb1.set_label('TEC / TECu', color=text_color)
	cb1.outline.set_edgecolor(edge_color)
	plt.setp(plt.getp(cb1.ax.axes, 'yticklabels'), color=text_color)

	ax1.coastlines(alpha=0.1)
	ax1.add_feature(cfeature.LAND, facecolor = landcolor, zorder=3, alpha=0.1)
	ax1.gridlines(alpha=0.3,draw_labels=True)
	ax1.xlabels_top = False
	ax1.ylabels_right = False
	ax1.xlines = False
	ax1.set_global()

	ax2 = the_plots[1,0]
	contour_plot2 = ax2.contourf(lonvals, latvals, nmf2vals, np.linspace(min_nmf2,max_nmf2,n_contours), extend='max', transform=ccrs.PlateCarree(central_longitude=0), cmap=cmap2)
	nmf2_title = ax2.text(0.5,1.05,'F2 Peak Electron Density (NmF2)',fontsize=title_size,transform=ax2.transAxes,horizontalalignment='center',color=text_color)

	cax2, kw = mpl.colorbar.make_axes(ax2,cmap=cmap2,pad=0.03,shrink=0.6)
	cb2=fig.colorbar(contour_plot2,cax=cax2,ticks=np.linspace(min_nmf2,max_nmf2,nticks),**kw)
	cb2.set_label('NmF2 / 1.e12 m-3', color=text_color)
	cb2.outline.set_edgecolor(edge_color)
	plt.setp(plt.getp(cb2.ax.axes, 'yticklabels'), color=text_color)

	ax2.coastlines(alpha=0.1)
	ax2.add_feature(cfeature.LAND, facecolor = landcolor, zorder=3, alpha=0.1)
	ax2.gridlines(alpha=0.3)
	ax2.set_global()

	ax3 = the_plots[1,1]
	contour_plot3 = ax3.contourf(lonvals, latvals, hmf2vals, np.linspace(200,1000,n_contours), extend='both', transform=ccrs.PlateCarree(central_longitude=0), cmap=cmap3)
	hmf2_title = ax3.text(0.5,1.05,'F2 Peak Height (hmF2)',fontsize=title_size,transform=ax3.transAxes,horizontalalignment='center',color=text_color)

	cax3, kw = mpl.colorbar.make_axes(ax3,cmap=cmap3,pad=0.03,shrink=0.6)
	cb3=fig.colorbar(contour_plot3,cax=cax3,ticks=np.linspace(200,1000,nticks),**kw)
	cb3.set_label('hmF2 / km', color=text_color)
	cb3.outline.set_edgecolor(edge_color)
	plt.setp(plt.getp(cb3.ax.axes, 'yticklabels'), color=text_color)

	ax3.coastlines(alpha=0.1)
	ax3.add_feature(cfeature.LAND, facecolor = landcolor, zorder=3, alpha=0.1)
	ax3.gridlines(alpha=0.3)
	ax3.set_global()

	plt.show()

if __name__ == '__main__':
	main()
