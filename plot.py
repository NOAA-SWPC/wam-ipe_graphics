from __future__ import print_function
import traceback
import matplotlib as mpl
mpl.use('Agg')
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
import matplotlib.colors as mcolors
from os.path import basename, exists

def get_full_archive_path(archive_path, path_fmt, dt):
    if   dt.hour < 3  or (dt.hour == 3  and dt.minute == 0):
        dt = dt.replace(hour=0)
    elif dt.hour < 9  or (dt.hour == 9  and dt.minute == 0):
        dt = dt.replace(hour=6)
    elif dt.hour < 15 or (dt.hour == 15 and dt.minute == 0):
        dt = dt.replace(hour=12)
    elif dt.hour < 21 or (dt.hour == 21 and dt.minute == 0):
        dt = dt.replace(hour=18)
    else:
        dt = dt.replace(hour=0) + timedelta(days=1)

    return '{}/{}'.format(archive_path, dt.strftime(path_fmt))

def get_archive_file(archive_path, path_fmt, file_fmt, dt):
    return '{}{}'.format(get_full_archive_path(archive_path, path_fmt, dt), dt.strftime(file_fmt))

def get_nday_median(archive_path, plot, opt, sdate, ndays=30, offset=0):
    var = plot['variable']
    path_fmt = plot['arch_path_fmt']
    file_fmt = plot['arch_file_fmt']
    dates = [sdate - timedelta(days=i) for i in range(1-offset,ndays+1-offset)]
    files = [get_archive_file(archive_path, path_fmt, file_fmt, dt) for dt in dates]
    if var == '__MUF__':
        data = [get_muf(Dataset(f), plot, opt) for f in files if exists(f)]
    else:
        data = [Dataset(f).variables[var][:] for f in files if exists(f)]
    return np.median(data, axis=0)

def replace_all(string, opt, nc_fid):
    for r in re.findall(r'__([^_].*?[^_])__',string):
        string = replace_text(string, r, opt, nc_fid)
    return string

def replace_text(string, replace_text, opt, nc_fid):
    if replace_text.lower() == "plot_vars":
        output = "-".join(
                      [plot['variable'].strip('_') for plot in opt['plots']
                      if plot['visible'] == True ])
    elif replace_text[-3:].lower() == '_ts':
        dt = datetime.strptime(nc_fid.getncattr(opt['{}_var'.format(replace_text.lower())]),
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
    vals *= np.sqrt(nmf2) / 1.11355287e+5
    return vals

def plot(file, opt, outpath='.', archive_path='.', archive_days=30):
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

    fig = plt.figure(figsize=opt['figsize'], facecolor=background_color)
    if opt['projection']['type'] == 'PlateCarree':
        if opt['projection']['rotating']:
            dt = datetime.strptime(nc_fid.getncattr(opt['projection']['rotating_var']),
                                   opt['metadata_ts_format'])
            central_longitude = -(dt.hour*60+dt.minute)*360/(60*24)
            proj = ccrs.PlateCarree(central_longitude=opt['projection']['central_longitude']+central_longitude)
        else:
            proj = ccrs.PlateCarree(central_longitude=opt['projection']['central_longitude'])
        tproj = ccrs.PlateCarree()
    elif opt['projection']['type'] == 'Orthographic':
        proj  = ccrs.NearsidePerspective(central_longitude=opt['projection']['central_longitude'],
                                         central_latitude =opt['projection']['central_latitude'])
#        tproj = ccrs.Geodetic()
        tproj = ccrs.PlateCarree()
    else:
        print('{} projection invalid!'.format(opt['projection']['type']))
        raise

    plots = fig.subplots(opt['shape'][0], opt['shape'][1],
                         subplot_kw=dict(projection=proj))
    if type(plots) == np.ndarray: plots = plots.flatten()
    else: plots = [plots]
    if opt['projection']['type'] == 'Orthographic':
        fig.subplots_adjust(left=-0.1, right=0.98, bottom=0.06, top=0.90, wspace=0.01, hspace=0.10)
    elif 'suptitle' in opt:
        fig.subplots_adjust(left=0.2, right=0.9, top=0.85)
    else:
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.95, wspace=0.1, hspace=0.15)

    # now make the plots
    for ax, plot in zip(plots, opt['plots']):
        # check if visible
        if not plot['visible']:
            ax.set_visible(False)
            continue

        try:
            try:
                if plot['use_arch']:
                    tfile = '{}/{}'.format(archive_path, basename(file))
            except:
                tfile = file
            nc_fid = Dataset(tfile, 'r', format='NETCDF4')
            lon = nc_fid.variables[opt['lonvar']][:]
            lat = nc_fid.variables[opt['latvar']][:]
            if 'trim_lats' in opt and opt['trim_lats']:
                lat = lat[1:-1]

            # get data and add cyclic point
            if plot['variable'] == "__MUF__":
                vals = get_muf(nc_fid, plot, opt)
            else:
                if 'level' in plot:
                    vals = nc_fid.variables[plot['variable']][plot['level']]
                else:
                    vals = nc_fid.variables[plot['variable']][:]

            try:
                if 'type' in plot:
                    cur_dt = datetime.strptime(nc_fid.getncattr(plot['ut_var']), opt['metadata_ts_format'])
                    start_dt = datetime.strptime(nc_fid.getncattr('init_date'), opt['metadata_ts_format'])
                    offset = (cur_dt-start_dt).days
                    if 'init_var' in plot:
                        init_dt = datetime.strptime(nc_fid.getncattr(plot['init_var']),opt['metadata_ts_format'])
                        offset = (cur_dt - init_dt).days
                    median_vals = get_nday_median(archive_path, plot, opt,
                                                  cur_dt, ndays=archive_days, offset=offset)
                    if plot['type'] == 'anomaly':
                        vals -= median_vals
                    elif plot['type'] == 'anomalypct':
                        vals = 100*(vals - median_vals)/median_vals
                    elif plot['type'] == 'median':
                        vals = median_vals
            except Exception as e:
                traceback.print_exc()
                pass

            vals *= plot['scale']
            if 'trim_lats' in opt and opt['trim_lats']:
                vals = vals[1:-1,:]
            vals, clon = cutil.add_cyclic_point(vals, coord=lon)

            # plot
            cmap = plt.get_cmap(plot['cmap'],256)
            contour_linspace = np.linspace(plot['min'],plot['max'],opt['ncontours'])
            tick_linspace    = np.linspace(plot['min'],plot['max'],plot['nticks'])
            contour_plot = ax.contourf(clon, lat, vals, contour_linspace,
                                        extend=plot['extend'], transform=tproj, cmap=cmap)

            # all the colorbar stuff
            cax, kw = mpl.colorbar.make_axes(ax,cmap=cmap,pad=0.03,shrink=0.6)
            cb=fig.colorbar(contour_plot,cax=cax,ticks=tick_linspace,**kw)
            cb.set_label(plot['cbar_label'], color=text_color, size=opt['cbar_label_size'],
                         fontfamily=opt['fontfamily'])
            cb.outline.set_edgecolor(edge_color)
            plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=text_color)

            # coastlines and continents
            ax.coastlines(alpha=opt['coastline_alpha'])
            ax.add_feature(cfeature.LAND, facecolor=land_color, zorder=3, alpha=opt['land_alpha'])

            # gridlines
            if 'gridline' in opt and opt['gridline']:
                gl = ax.gridlines(alpha=opt['gridline_alpha'],draw_labels=True)
                gl.top_labels = False
                gl.right_labels = False
                gl.xlines = False
                gl.ylines = False
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
                label_style = {'size': opt['axis_label_size'], 'color': text_color,
                               'fontfamily': opt['fontfamily']}
                gl.xlabel_style = label_style
                gl.ylabel_style = label_style

            if opt['projection']['type'] == 'Orthographic':
                ax.set_global()

            if 'extent' in plot:
                ax.set_extent(plot['extent'], crs=tproj)

            # plot title
            ax.set_title(plot['title'], fontsize=opt['title_size'], color=text_color,
                         fontfamily=opt['fontfamily'])
            #print('done {}'.format(i))
        except Exception as e:
#            print('Error while drawing plot {}'.format(i))
#            traceback.print_exc()
#            print(e)
            pass

    # add text if the plot definition includes it
    if 'texts' in opt:
        for text in opt['texts']:
            output_string = replace_all(text['text'], opt, nc_fid)
            fig.text(text['x_loc'], text['y_loc'], output_string,
                     fontsize=text['size'], horizontalalignment=text['align'],
                     color=text_color, fontfamily=opt['fontfamily'])

    if 'suptitle' in opt:
        output_string = replace_all(opt['suptitle'], opt, nc_fid)
        fig.suptitle(output_string, fontsize=24, color=text_color)

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
    parser.add_argument('-d', '--archdays', help='days to search backwards for averaging', type=int, default=30)
    args = parser.parse_args()

    try:
        opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
        if args.path != "":
            with Pool(processes=args.tasks) as p:
                files = glob.glob("{}/{}*.nc".format(args.path, args.prefix))
                p.starmap(plot,zip(files, itertools.repeat(opt), itertools.repeat(args.outpath), itertools.repeat(args.archpath), itertools.repeat(args.archdays)))
        else:
            plot(args.file, opt, args.outpath, args.archpath, args.archdays)
    except Exception as e:
        print(e)
        pass

if __name__ == '__main__':
    main()
