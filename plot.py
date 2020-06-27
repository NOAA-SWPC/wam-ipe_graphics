from __future__ import print_function
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import glob
import cartopy
import cartopy.crs as ccrs
import cartopy.util as cutil
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from datetime import datetime
from multiprocessing import Pool
import numpy as np
import yaml
import re
import itertools
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def replace_all(string, opt, nc_fid):
    for r in re.findall(r'__([^_].*?[^_])__',string):
        string = replace_text(string, r, opt, nc_fid)
    return string

def replace_text(string, replace_text, opt, nc_fid):
    if replace_text[-3:] == '_ts':
        dt = datetime.strptime(nc_fid.getncattr(opt['{}_var'.format(replace_text)]), \
                         opt['metadata_ts_format'])
        output = dt.strftime(opt['{}_format'.format(replace_text)])
    else:
        output = nc_fid.getncattr(opt['{}_var'.format(replace_text)])
    return string.replace('__{}__'.format(replace_text), output)

def plot(file, opt, outpath='.'):

    if opt['scheme'] == 'light':
        text_color       = (0.0, 0.0, 0.0)
        edge_color       = (0.0, 0.4, 0.0)
        background_color = (0.9, 0.9, 0.9)
        land_color       = (0.0, 0.0, 0.0)
    elif opt['scheme'] == 'dark':
        text_color       = (0.9, 0.9, 0.9)
        edge_color       = (0.0, 0.4, 0.0)
        background_color = (0.1, 0.1, 0.1)
        land_color       = (1.0, 1.0, 1.0)
    else: # default to light
        text_color       = (0.0, 0.0, 0.0)
        edge_color       = (0.0, 0.4, 0.0)
        background_color = (0.9, 0.9, 0.9)
        land_color       = (0.0, 0.0, 0.0)

    print(file)
    nc_fid = Dataset(file, 'r', format='NETCDF4')

    lon = nc_fid.variables[opt['lonvar']][:]
    lat = nc_fid.variables[opt['latvar']][:]

    fig = plt.figure(figsize=opt['figsize'], facecolor=background_color)
    if opt['projection']['type'] == 'PlateCarree':
        if opt['projection']['rotating']:
            dt = datetime.strptime(nc_fid.getncattr(opt['projection']['rotating_var']),opt['metadata_ts_format'])
            central_longitude = -(dt.hour*60+dt.minute)*360/(60*24)
            proj = ccrs.PlateCarree(central_longitude=opt['projection']['central_longitude']+central_longitude)
        else:
            proj = ccrs.PlateCarree(central_longitude=opt['projection']['central_longitude'])
        tproj = ccrs.PlateCarree()
    else:
        print('{} projection invalid!'.format(opt['projection']['type']))
        raise

    plots = fig.subplots(opt['shape'][0], opt['shape'][1], \
                         subplot_kw=dict(projection=proj)).flatten()
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.95, wspace=0.1, hspace=0.15)

    # now make the plots
    for i, plot in enumerate(opt['plots']):

        # get axis object
        ax = plots[i]

        # check if visible
        if not plot['visible']:
            ax.set_visible(False)
            continue

        try:
            # get data and add cyclic point
            vals = nc_fid.variables[plot['variable']][:] * plot['scale']
            vals, clon = cutil.add_cyclic_point(vals, coord=lon)

            # plot
            cmap = plt.get_cmap(plot['cmap'],256)
            contour_linspace = np.linspace(plot['min'],plot['max'],opt['ncontours'])
            tick_linspace    = np.linspace(plot['min'],plot['max'],plot['nticks'])
            contour_plot = ax.contourf(clon, lat, vals, contour_linspace, \
                                        extend=plot['extend'], transform=tproj, cmap=cmap)

            # all the colorbar stuff
            cax, kw = mpl.colorbar.make_axes(ax,cmap=cmap,pad=0.03,shrink=0.6)
            cb=fig.colorbar(contour_plot,cax=cax,ticks=tick_linspace,**kw)
            cb.set_label(plot['cbar_label'], color=text_color, size=opt['cbar_label_size'])
            cb.outline.set_edgecolor(edge_color)
            plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=text_color)

            # coastlines and continents
            ax.coastlines(alpha=0.1)
            ax.add_feature(cfeature.LAND, facecolor=land_color, zorder=3, alpha=0.1)

            # gridlines
            gl = ax.gridlines(alpha=0.3,draw_labels=True)
            gl.xlabels_top = False
            gl.ylabels_right = False
            gl.xlines = False
            gl.ylines = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': opt['axis_label_size'], 'color': text_color}
            gl.ylabel_style = {'size': opt['axis_label_size'], 'color': text_color}

            #ax.set_global()

            # plot title
            ax.set_title(plot['title'], fontsize=opt['title_size'], color=text_color)

        except Exception as e:
            print('Error while drawing plot {}'.format(i))
            print(e)
            pass

    # add text if the plot definition includes it
    if 'texts' in opt:
        for text in opt['texts']:
            output_string = replace_all(text['text'], opt, nc_fid)
            fig.text(text['x_loc'],text['y_loc'],output_string,fontsize=text['size'],horizontalalignment=text['align'],color=text_color)

#    plt.show()
    plt.savefig("{}/{}".format(outpath,replace_all(opt['output_format'], opt, nc_fid)), facecolor=background_color)
    plt.close()

def main():
    parser = ArgumentParser(description='plot tec, nmf2, hmf2', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config',  help='yaml config', type=str, required=True)
    parser.add_argument('-f', '--file',    help='filename', type=str, default='')
    parser.add_argument('-p', '--path',    help='path to files (also requires -r, overrides -f)', type=str, default='')
    parser.add_argument('-r', '--prefix',  help='file prefix (also requires -p) -- will plot all path/prefix*.nc files', type=str, default='')
    parser.add_argument('-t', '--tasks',   help='parallel plotting tasks (only used with -p)', type=int, default=1)
    parser.add_argument('-o', '--outpath', help='output path', type=str, default='.')
    args = parser.parse_args()

    try:
        opt = yaml.load(open(args.config),Loader=yaml.FullLoader)
        if args.path != "":
            with Pool(processes=args.tasks) as p:
                files = glob.glob("{}/{}*.nc".format(args.path,args.prefix))
                p.starmap(plot,zip(files, itertools.repeat(opt), itertools.repeat(args.outpath)))
        else:
            plot(args.file, opt, args.outpath)
    except Exception as e:
        print(e)
        pass

if __name__ == '__main__':
    main()
