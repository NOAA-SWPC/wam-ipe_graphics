from __future__ import print_function
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import glob
from math import pi
import cartopy
import cartopy.crs as ccrs
import cartopy.util as cutil
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from datetime import datetime, timedelta
from multiprocessing import Pool
import numpy as np
import yaml
import re
import itertools
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def get_full_archive_path(archive_path, dt, formatter):
    search_dt = dt.replace(minute=0, second=0)

    if   search_dt.hour < 3:
        search_dt = search_dt.replace(hour=0)
    elif search_dt.hour < 9:
        search_dt = search_dt.replace(hour=6)
    elif search_dt.hour < 15:
        search_dt = search_dt.replace(hour=12)
    elif search_dt.hour < 21:
        search_dt = search_dt.replace(hour=18)
    else:
        search_dt = search_dt.replace(day=search_dt.day+1, hour=0)

    return '{}/{}'.format(archive_path, search_dt.strftime(formatter))

def get_n_day_mean(archive_path, plot, opt, dt, days=30):
    variable = plot['variable']
    archive_path_fmt = plot['arch_path_fmt']
    archive_file_fmt = plot['arch_file_fmt']

    for td in range(1,days+1):
        search_dt = dt - timedelta(days=td)
        path = get_full_archive_path(archive_path, search_dt, archive_path_fmt)
        file = '{}{}'.format(path, search_dt.strftime(archive_file_fmt))
        try:
            ds = Dataset(file)
        except:
            pass
        if td == 1:
            if variable == '__MUF__':
                sum_data = get_muf(ds, plot, opt)
            else:
                sum_data = ds.variables[variable][:]
            count = 1
            ds.close()
        else:
            try:
                if variable == '__MUF__':
                    sum_data += get_muf(ds, plot, opt)
                else:
                    sum_data += ds.variables[variable][:]
                count += 1
                ds.close()
            except:
                pass
    print('count',count)
    return sum_data/count

def replace_all(string, opt, nc_fid):
    for r in re.findall(r'__([^_].*?[^_])__',string):
        string = replace_text(string, r, opt, nc_fid)
    return string

def replace_text(string, replace_text, opt, nc_fid):
    if replace_text.lower() == "plot_vars":
        output = "-".join( \
                      [plot['variable'].strip('_') for plot in opt['plots'] \
                      if plot['visible'] == True ])
    elif replace_text[-3:].lower() == '_ts':
        dt = datetime.strptime(nc_fid.getncattr(opt['{}_var'.format(replace_text.lower())]), \
                         opt['metadata_ts_format'])
        output = dt.strftime(opt['{}_format'.format(replace_text.lower())])
    else:
        output = nc_fid.getncattr(opt['{}_var'.format(replace_text.lower())])

    if replace_text.isupper():
        return string.replace('__{}__'.format(replace_text), output.upper())
    else:
        return string.replace('__{}__'.format(replace_text), output)

def get_muf(nc_fid, plot, opt):
    lon = nc_fid.variables[opt['lonvar']][:]
    hmf2 = nc_fid.variables[plot['hmf2_var']][:]
    nmf2 = nc_fid.variables[plot['nmf2_var']][:]
    ut = datetime.strptime(nc_fid.getncattr(plot['ut_var']), opt['metadata_ts_format'])
    loctime = ( (ut.hour + ut.minute / 60) + lon / 15 ) % 24
    vals  = 1490 / (hmf2 + 176)
    vals -= 0.6*np.sin((loctime-5) * pi/12)
    vals *= np.sqrt(nmf2) * 1.11355287e-5
    return vals

def plot(file, opt, outpath='.', archive_path='.'):

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
            dt = datetime.strptime(nc_fid.getncattr(opt['projection']['rotating_var']), \
                                   opt['metadata_ts_format'])
            central_longitude = -(dt.hour*60+dt.minute)*360/(60*24)
            proj = ccrs.PlateCarree(central_longitude=opt['projection']['central_longitude']+central_longitude)
        else:
            proj = ccrs.PlateCarree(central_longitude=opt['projection']['central_longitude'])
        tproj = ccrs.PlateCarree()
    else:
        print('{} projection invalid!'.format(opt['projection']['type']))
        raise

    plots = fig.subplots(opt['shape'][0], opt['shape'][1], \
                         subplot_kw=dict(projection=proj))
    if type(plots) == np.ndarray: plots = plots.flatten()
    else: plots = [plots]
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
            if plot['variable'] == "__MUF__":
                vals = get_muf(nc_fid, plot, opt)
            else:
                vals = nc_fid.variables[plot['variable']][:]

            try:
                if plot['type'] == 'anomaly':
                    print('anomaly')
                    mean_vals = get_n_day_mean(archive_path, plot, opt,
                                       datetime.strptime(nc_fid.getncattr(plot['ut_var']), opt['metadata_ts_format']))
                    print(mean_vals)
                    vals -= mean_vals
                elif plot['type'] == 'mean':
                    print('mean')
                    mean_vals = get_n_day_mean(archive_path, plot, opt,
                                       datetime.strptime(nc_fid.getncattr(plot['ut_var']), opt['metadata_ts_format']))
                    vals = mean_vals
            except:
                pass

            vals *= plot['scale']
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
            cb.set_label(plot['cbar_label'], color=text_color, size=opt['cbar_label_size'], \
                         fontfamily=opt['fontfamily'])
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
            label_style = {'size': opt['axis_label_size'], 'color': text_color, \
                           'fontfamily': opt['fontfamily']}
            gl.xlabel_style = label_style
            gl.ylabel_style = label_style

            #ax.set_global()

            # plot title
            ax.set_title(plot['title'], fontsize=opt['title_size'], color=text_color, \
                         fontfamily=opt['fontfamily'])
            print('done {}'.format(i))
        except Exception as e:
            print('Error while drawing plot {}'.format(i))
            print(e)
            pass

    # add text if the plot definition includes it
    if 'texts' in opt:
        for text in opt['texts']:
            output_string = replace_all(text['text'], opt, nc_fid)
            fig.text(text['x_loc'], text['y_loc'], output_string, \
                     fontsize=text['size'], horizontalalignment=text['align'], \
                     color=text_color, fontfamily=opt['fontfamily'])

#    plt.show()
    filename = [s.upper() for s in replace_all(opt['output_format'], opt, nc_fid).split(".")]
    filename[-1] = filename[-1].lower()
    filename = ".".join(filename)
    plt.savefig("{}/{}".format(outpath,filename), facecolor=background_color)
    plt.close()

def main():
    parser = ArgumentParser(description='plot tec, nmf2, hmf2', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config',   help='yaml config', type=str, required=True)
    parser.add_argument('-f', '--file',     help='filename', type=str, default='')
    parser.add_argument('-p', '--path',     help='path to files (also requires -r, overrides -f)', type=str, default='')
    parser.add_argument('-r', '--prefix',   help='file prefix (used with -p) -- will plot all path/prefix*.nc files', type=str, default='')
    parser.add_argument('-t', '--tasks',    help='parallel plotting tasks (only used with -p)', type=int, default=1)
    parser.add_argument('-o', '--outpath',  help='output path', type=str, default='.')
    parser.add_argument('-a', '--archpath', help='archive path for anomaly plots', type=str, default='.')
    args = parser.parse_args()

    try:
        opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
        if args.path != "":
            with Pool(processes=args.tasks) as p:
                files = glob.glob("{}/{}*.nc".format(args.path, args.prefix))
                p.starmap(plot,zip(files, itertools.repeat(opt), itertools.repeat(args.outpath), itertools.repeat(args.archpath)))
        else:
            plot(args.file, opt, args.outpath, args.archpath)
    except Exception as e:
        print(e)
        pass

if __name__ == '__main__':
    main()
