#diskcache to speed it up needed

#pip install diskcache
#pip install -e ../

import os
from pathlib import Path
from datetime import datetime, timedelta
from io import StringIO
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# viz and ui
import hvplot.pandas
import holoviews as hv
from holoviews import opts, dim, streams
hv.extension('bokeh')
import cartopy
import geoviews as gv
gv.extension('bokeh')
import panel as pn
pn.extension('tabulator')
import param
#
from vtools.functions.filter import godin

# this should be a util function
def find_lastest_fname(pattern, dir='.'):
    d = Path(dir)
    fname,mtime=None,0
    for f in d.glob(pattern):
        fmtime = f.stat().st_mtime
        if fmtime > mtime:
            mtime = fmtime
            fname = f.absolute()
    return fname, mtime

#!pip install diskcache
import diskcache
cache = diskcache.Cache('./cache', size_limit=1e11)
import dms_datastore
from dms_datastore.read_ts import read_ts

@cache.memoize()
def get_station_data_for_filename(filename, directory):
    return read_ts(os.path.join(directory,filename))

# from stackoverflow.com https://stackoverflow.com/questions/6086976/how-to-get-a-complete-exception-stack-trace-in-python
def full_stack():
    import traceback, sys
    exc = sys.exc_info()[0]
    stack = traceback.extract_stack()[:-1]  # last one would be full_stack()
    if exc is not None:  # i.e. an exception is present
        del stack[-1]       # remove call of full_stack, the printed exception
                            # will contain the caught exception caller instead
    trc = 'Traceback (most recent call last):\n'
    stackstr = trc + ''.join(traceback.format_list(stack))
    if exc is not None:
         stackstr += '  ' + traceback.format_exc().lstrip(trc)
    return stackstr

from bokeh.models import HoverTool

class StationInventoryExplorer(param.Parameterized):
    '''
    Show station inventory on map and select to display data available
    Furthermore select the data rows and click on button to display plots for selected rows
    '''
    time_window = param.CalendarDateRange(default=(datetime.now()- timedelta(days=10), datetime.now()))
    repo_level = param.Selector(objects=['formatted_1yr', 'screened'], default='formatted_1yr',
                                doc='repository level (sub directory) under which data is found')
    parameter_type = param.ListSelector(objects=['all'],
                                    default=['all'],
                                    doc='parameter type'
                                    )
    show_legend = param.Boolean(default=True)
    godin_filter = param.Boolean(default=False, doc='Apply Godin filter to data')

    def __init__(self, dir, **kwargs):
        super().__init__(**kwargs)
        self.dir = dir
        self.inventory_file, mtime = find_lastest_fname(
            'inventory*.csv', self.dir)
        self.df_dataset_inventory = pd.read_csv(
            os.path.join(self.dir, self.inventory_file))
        # replace nan with empty string for column subloc
        self.df_dataset_inventory['subloc'] = self.df_dataset_inventory['subloc'].fillna('')
        self.param.parameter_type.objects = ['all'] +  list(self.df_dataset_inventory['param'].unique())
        self.map = self.df_dataset_inventory.hvplot.points(x='lon', y='lat', by='agency',
                                                           tiles='CartoLight', geo=True, projection=cartopy.crs.GOOGLE_MERCATOR,
                                                           hover_cols='all', s=35).opts(active_tools=['wheel_zoom'])
        group_cols = ['station_id', 'subloc', 'name', 'unit', 'param',
                      'min_year', 'max_year', 'agency', 'agency_id_dbase', 'lat', 'lon']
        self.df_station_inventory = self.df_dataset_inventory.groupby(
            group_cols).count().reset_index()[group_cols]
        self.tmap = gv.tile_sources.CartoLight
        tooltips = [
            ('Station ID', '@station_id'),
            ('SubLoc', '@subloc'),
            ('Name', '@name'),
            ('Years', '@min_year to @max_year'),
            ('Agency', '@agency - @agency_id_dbase'),
            ('Parameter', '@param'),
            ('Unit', '@unit')
        ]
        hover = HoverTool(tooltips=tooltips)
        self.current_station_inventory = self.df_station_inventory
        self.map_station_inventory = gv.Points(self.current_station_inventory, kdims=['lon', 'lat']
                                              ).opts(size=6, color=dim('param'), cmap='Category10', tools=[hover], height=800)
        self.map_station_inventory = self.map_station_inventory.opts(opts.Points(tools=['tap', hover], #'hover'],
                                                                                 nonselection_alpha=0.3,  # nonselection_color='gray',
                                                                                 size=8)
                                                                     ).opts(frame_width=500, active_tools=['wheel_zoom'])

        self.station_select = streams.Selection1D(source=self.map_station_inventory)#.Points.I)

    def show_inventory(self, index):
        if len(index) == 0:
            index = [0]
        dfs = self.current_station_inventory.iloc[index]
        # return a UI with controls to plot and show data
        return self.update_data_table(dfs)

    def save_dataframe(self, event):
        df = self.display_table.value.iloc[self.display_table.selection]
        df = df.merge(self.df_dataset_inventory)
        for i, r in df.iterrows():
            dfdata = self.get_data_for(r)
            param = r['param']
            unit = r['unit']
            station_id = r['station_id']
            agency = r['agency']
            #agency_id_dbase = r['agency_id_dbase']
            dfdata.to_csv(f'saved_{agency}_{station_id}_{param}.csv')

    def create_plots(self, event):
        #df = self.display_table.selected_dataframe # buggy
        df = self.display_table.value.iloc[self.display_table.selection]
        df = df.merge(self.df_dataset_inventory)
        try:
            layout_map = {}
            for i, r in df.iterrows():
                crv = self.create_curve(r)
                unit = r['unit']
                if unit not in layout_map:
                    layout_map[unit] = []
                layout_map[unit].append(crv)
            if len(layout_map) == 0:
                return hv.Div('<h3>Select rows from table and click on button</h3>')
            else:
                return hv.Layout([hv.Overlay(layout_map[k]) for k in layout_map]).cols(1).opts(shared_axes=False)
        except Exception as e:
            stackmsg = full_stack()
            print(stackmsg)
            return hv.Div(f'<h3> Exception while fetching data </h3> <pre>{stackmsg}</pre>')

    def create_curve(self, r):
        filename = r['filename']
        param = r['param']
        unit = r['unit']
        station_id = r['station_id']
        agency = r['agency']
        agency_id_dbase = r['agency_id_dbase']
        df = self.get_data_for(r)
        if self.godin_filter:
            df['value'] = godin(df['value'])
        crv = hv.Curve(df[['value']]).redim(value=f'{station_id}/{param}', datetime='Time')
        return crv.opts(ylabel=f'{param}({unit})', title=f'{station_id}::{agency}/{agency_id_dbase}', show_legend=self.show_legend, responsive=True, active_tools=['wheel_zoom'], tools=['hover'])

    def get_data_for(self, r):
        filename = r['filename']
        param = r['param']
        unit = r['unit']
        station_id = r['station_id']
        agency = r['agency']
        agency_id_dbase = r['agency_id_dbase']
        df = get_station_data_for_filename(os.path.join(self.repo_level, filename), self.dir)
        df = df.loc[slice(*self.time_window), :]
        return df

    def cache(self):
        # get unique filenames
        filenames = self.df_dataset_inventory['filename'].unique()
        print('Caching: ', len(filenames), ' files')
        for i, filename in enumerate(filenames):
            print(f'Caching {i} ::{filename}')
            get_station_data_for_filename(os.path.join(self.repo_level, filename), self.dir)

    def update_plots(self, event):
        self.plot_panel.loading = True
        self.plot_panel.object = self.create_plots(event)
        self.plot_panel.loading = False

    def download_data(self):
        df = self.display_table.value.iloc[self.display_table.selection]
        df = df.merge(self.df_dataset_inventory)
        dflist = []
        for i, r in df.iterrows():
            dfdata = self.get_data_for(r)
            param = r['param']
            unit = r['unit']
            subloc = r['subloc']
            station_id = r['station_id']
            agency = r['agency']
            agency_id_dbase = r['agency_id_dbase']
            dfdata.columns = [f'{station_id}/{subloc}/{agency}/{agency_id_dbase}/{param}/{unit}']
            dflist.append(dfdata)
        dfdata = pd.concat(dflist, axis=1)
        sio = StringIO()
        dfdata.to_csv(sio)
        sio.seek(0)
        return sio

    def update_data_table(self, dfs):
        # if attribute display_table is not set, create it
        if not hasattr(self, 'display_table'):
            column_width_map = {'index': '5%', 'station_id': '10%', 'subloc': '5%', 'lat': '5%', 'lon': '5%', 'name': '25%',
                                'min_year': '5%', 'max_year':'5%', 'agency': '5%', 'agency_id_dbase': '5%', 'param': '5%', 'unit': '5%'}
            self.display_table = pn.widgets.Tabulator(dfs, disabled=True, widths=column_width_map)
            self.plot_button = pn.widgets.Button(name="Plot Selected", button_type="primary")
            self.plot_button.on_click(self.update_plots)
            self.plot_panel = pn.panel(hv.Div('<h3>Select rows from table and click on button</h3>'))
            # add a button to trigger the save function
            self.download_button = pn.widgets.FileDownload(callback=self.download_data, filename='dms_data.csv', button_type='success', embed=False)
            gspec = pn.GridSpec(sizing_mode='stretch_both', max_height=3600, max_width=1600)
            gspec[0,:3] = pn.Row(self.plot_button, self.download_button)
            gspec[1:3,0:10] = self.display_table
            gspec[3:10,0:10] = self.plot_panel
            self.plots_panel = pn.Row(gspec) # fails with object of type 'GridSpec' has no len()
        else:
            self.display_table.value = dfs
        return self.plots_panel

    def get_map_of_stations(self, vartype):
        if len(vartype)==1 and vartype[0] == 'all':
            dfs = self.df_station_inventory
        else:
            dfs = self.df_station_inventory[self.df_station_inventory['param'].isin(vartype)]
        self.current_station_inventory = dfs
        self.map_station_inventory.data = self.current_station_inventory
        return self.tmap*self.map_station_inventory

    def create_maps_view(self):
        control_widgets = pn.Param(self, widgets={"time_window": pn.widgets.DatetimeRangePicker})
        col1 = pn.Column(control_widgets, pn.bind(self.get_map_of_stations, vartype=self.param.parameter_type), width=600)
        col2 = pn.Column(pn.bind(self.show_inventory, index=self.station_select.param.index))
        return pn.Row(col1, col2, sizing_mode='stretch_both')

#!conda install -y -c conda-forge jupyter_bokeh
if __name__ == '__main__':
    # using argparse to get the directory
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='directory with station inventory')
    # add argument optional to run caching
    parser.add_argument('--cache', help='use caching', action='store_true')
    args = parser.parse_args()
    # if no args exit with help message
    if args.dir is None:
        parser.print_help()
        exit(0)
    else:
        dir = args.dir
    explorer = StationInventoryExplorer(dir)
    if args.cache:
        # run caching
        print('Clearing cache')
        cache.clear()
        print('Caching data')
        explorer.cache()
    else: # display ui
        explorer.create_maps_view().show(title='Station Inventory Explorer')
