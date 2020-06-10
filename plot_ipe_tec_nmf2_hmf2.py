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
            print "\t\ttype:", repr(nc_fid.variables[key].dtype)
            for ncattr in nc_fid.variables[key].ncattrs():
                print '\t\t%s:' % ncattr,\
                      repr(nc_fid.variables[key].getncattr(ncattr))
        except KeyError:
            print "\t\tWARNING: %s does not contain variable attributes" % key

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print "NetCDF Global Attributes:"
        for nc_attr in nc_attrs:
            print '\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print "NetCDF dimension information:"
        for dim in nc_dims:
            print "\tName:", dim 
            print "\t\tsize:", len(nc_fid.dimensions[dim])
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print "NetCDF variable information:"
        for var in nc_vars:
            if var not in nc_dims:
                print '\tName:', var
                print "\t\tdimensions:", nc_fid.variables[var].dimensions
                print "\t\tsize:", nc_fid.variables[var].size
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars


def main():

    input_output_path = "/Users/georgemillward/Downloads/"

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=0))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0., top=0.9)

    # nc_fid = Dataset("/Users/georgemillward/Downloads/ipe.20150317_151500.nc", "r", format="NETCDF4")
    nc_fid = Dataset(input_output_path+"ipe.20130316_010000_nmf2.nc", "r", format="NETCDF4")
    nc_attrs, nc_dims, nc_vars = ncdump(nc_fid)
    for var in nc_vars:
        if var not in nc_dims:
            print '\tName:', var
            print "\t\tdimensions:", nc_fid.variables[var].dimensions
            print "\t\tsize:", nc_fid.variables[var].size
            #print_ncattr(var)
    lon = nc_fid.variables['lon'] # longitude
    lat = nc_fid.variables['lat'] # latitude
    tec = nc_fid.variables['tec'] # TEC
    nmf2 = nc_fid.variables['nmf2'] # nmf2
    hmf2 = nc_fid.variables['hmf2'] # nmf2
    
    latvals = lat[:]
    print 'latvals'
    print latvals

    lonvals = lon[:]
    print 'lonvals'
    print lonvals

    tecvals = tec[:]
    print 'tecvals'
    print tecvals

    nmf2vals = nmf2[:]
    hmf2vals = hmf2[:]

    cmap = plt.get_cmap('cividis',256)
    cmap2 = plt.get_cmap('viridis',256)
    cmap3 = plt.get_cmap('jet',256)

    contour_plot = ax.contourf(lonvals, latvals, tecvals, 20, transform=ccrs.PlateCarree(central_longitude=0), cmap=cmap)

    ax2 = fig.add_subplot(3,20,60)

    cb = mpl.colorbar.ColorbarBase(ax2,cmap=cmap,orientation='vertical')
    cb.set_label('TEC (TECu)')  

    ax.coastlines(alpha=0.2)
    ax.gridlines(alpha=0.1)

    # ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black', color='black')
    ax.set_global()
    tec_title = ax.text(0.5,1.05,'WAM-IPE Total Electron Content TEC (TECu)',fontsize=16,transform=ax.transAxes,horizontalalignment='center')

    # plt.show()

    plt.savefig(input_output_path+"tec.png")
    tec_title.remove()
    # contour_plot.remove()

# Now the nmf2 plot....

    contour_plot = ax.contourf(lonvals, latvals, nmf2vals,20,
        transform=ccrs.PlateCarree(),
        cmap=cmap2)
    cb = mpl.colorbar.ColorbarBase(ax2,cmap=cmap2,orientation='vertical')
    cb.set_label('NmF2') 

    nmf2_title = ax.text(0.5,1.05,'WAM-IPE nmf2 (m-3)',fontsize=16,transform=ax.transAxes,horizontalalignment='center')
    plt.savefig(input_output_path+"nmf2.png")
    nmf2_title.remove()
    # contour_plot.remove()

# Now the hmf2 plot....

    contour_plot = ax.contourf(lonvals, latvals, hmf2vals,20,
        transform=ccrs.PlateCarree(),
        cmap=cmap3)
    cb = mpl.colorbar.ColorbarBase(ax2,cmap=cmap3,orientation='vertical')
    cb.set_label('hmF2')
    hmf2_title = ax.text(0.5,1.05,'WAM-IPE hmf2 (km)',fontsize=16,transform=ax.transAxes,horizontalalignment='center')
    plt.savefig(input_output_path+"hmf2.png")


# Calculate MUF(3000)
# loctime = Local Time (0-23:59 corresponding to the longitude) This changes with UT!
# nlongs = number of longitudes
# nlats = number of latitudes
# for i=0, nlongs do begin
# for j=0, nlats do begin
# M3000(i,j) = 1490./(hmF2(i,j) + 176.)
# M3000(i,j) = M3000(i,j) - .6sin(3.1415/12.(loctime(i) - 5.))
# MUF3000(i,j) = M3000(i,j)*sqrt(nmF2(i,j))/(1.11355287e+5)
# endfor ; j=0,90
# endfor ;i=0,20


if __name__ == '__main__':
    main()
