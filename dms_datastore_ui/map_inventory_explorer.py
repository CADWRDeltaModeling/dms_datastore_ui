#diskcache to speed it up needed

#pip install diskcache
#pip install -e ../

import os
from pathlib import Path
from datetime import datetime
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
cache = diskcache.Cache('./cache')
import dms_datastore
from dms_datastore.read_ts import read_ts

@cache.memoize(expire=3600)
def get_station_data_for_filename(filename, directory):
    return read_ts(os.path.join(directory,filename))

from bokeh.models import HoverTool

class StationInventoryExplorer(param.Parameterized):
    '''
    Show station inventory on map and select to display data available
    Furthermore select the data rows and click on button to display plots for selected rows
    '''
    time_window = param.String(default='10D', regex='\d+[H|D]',
                               doc='timewindow from end of data in hours(H) or days (D)')
    repo_level = param.Selector(objects=['formatted', 'screened'], default='formatted',
                                doc='repository level (sub directory) under which data is found')
    parameter_type = param.Selector(objects=['all', 'cla', 'do', 'ec', 'elev', 'fdom', 'flow',
                                     'ph', 'predictions', 'salinity', 'ssc', 'temp', 'turbidity', 'velocity'],
                                    default='all',
                                    doc='parameter type'
                                    )
    show_legend = param.Boolean(default=True)

    def __init__(self, dir, **kwargs):
        super().__init__(**kwargs)
        self.dir = dir
        self.inventory_file, mtime = find_lastest_fname(
            'inventory*.csv', self.dir)
        self.df_dataset_inventory = pd.read_csv(
            os.path.join(self.dir, self.inventory_file))
        self.map = self.df_dataset_inventory.hvplot.points(x='lon', y='lat', by='agency',
                                                           tiles='CartoLight', geo=True, projection=cartopy.crs.GOOGLE_MERCATOR,
                                                           hover_cols='all', s=35).opts(active_tools=['wheel_zoom'])
        group_cols = ['station_id', 'lat', 'lon', 'name',
                      'min_year', 'max_year', 'agency', 'unit', 'param']
        self.df_station_inventory = self.df_dataset_inventory.groupby(
            group_cols).count().reset_index()[group_cols]
        self.tmap = gv.tile_sources.CartoLight
        tooltips = [
            ('Station ID', '@station_id'),
            ('Name', '@name'),
            ('Years', '@min_year to @max_year'),
            ('Agency', '@agency'),
            ('Parameter', '@param')
        ]
        hover = HoverTool(tooltips=tooltips)
        self.current_station_inventory = self.df_station_inventory
        self.map_station_inventory = gv.Points(self.current_station_inventory, kdims=['lon', 'lat']
                                              ).opts(size=6, color=dim('param'), cmap='Category10', tools=[hover], height=800)
        # self.map_station_inventory = self.df_station_inventory.hvplot.points(x='lon', y='lat',
        #                                                                     tiles='CartoLight', geo=True,
        #                                                                     by='param',
        #                                                                     hover_cols=['station_id', 'name'])
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
            agency_dbase = r['agency_dbase']
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
                return hv.Layout([hv.Overlay(layout_map[k]) for k in layout_map]
                                 ).cols(1).opts(shared_axes=False)
        except Exception as e:
            print(e)
            breakpoint
            return hv.Div(f'<h3> Exception while fetching data </h3> <pre>{e}</pre>')

    def create_curve(self, r):
        filename = r['filename']
        param = r['param']
        unit = r['unit']
        station_id = r['station_id']
        agency = r['agency']
        agency_dbase = r['agency_dbase']
        df = self.get_data_for(r)
        crv = hv.Curve(df[['value']]).redim(
            value=f'{station_id}', datetime='Time').opts(width=600, tools=['hover'])
        return crv.opts(ylabel=f'{param}({unit})', title=f'{station_id}::{agency}/{agency_dbase}', show_legend=self.show_legend)

    def get_data_for(self, r):
        filename = r['filename']
        param = r['param']
        unit = r['unit']
        station_id = r['station_id']
        agency = r['agency']
        agency_dbase = r['agency_dbase']
        df = get_station_data_for_filename(
            f'{self.repo_level}\\{filename}', self.dir)
        df = df.loc[slice(
            df.index[-1]-pd.Timedelta(self.time_window), df.index[-1]), :]
        return df

    def update_plots(self, event):
        self.plot_panel.loading = True
        self.plot_panel.object = self.create_plots(event)
        self.plot_panel.loading = False

    def update_data_table(self, dfs):
        # if attribute display_table is not set, create it
        if not hasattr(self, 'display_table'):
            self.display_table = pn.widgets.Tabulator(dfs, disabled=True)
            self.plot_button = pn.widgets.Button(name="Plot Selected", button_type="primary")
            self.plot_button.on_click(self.update_plots)
            self.plot_panel = pn.panel(hv.Div('<h3>Select rows from table and click on button</h3>'))
            # add a button to trigger the save function
            self.save_button = pn.widgets.Button(name="Save DataFrame", button_type="primary")
            self.save_button.on_click(self.save_dataframe)
            self.plots_panel = pn.Column(self.display_table, self.plot_button, self.save_button, self.plot_panel)
        else:
            self.display_table.value = dfs
        return self.plots_panel

    def get_map_of_stations(self, vartype):
        dfs = self.df_station_inventory
        if vartype != 'all':
            dfs = self.df_station_inventory.query(f'param == "{vartype}"')
        self.current_station_inventory = dfs
        self.map_station_inventory.data = self.current_station_inventory
        return self.tmap*self.map_station_inventory

    def create_maps_view(self):
        col1 = pn.Column(self.param, pn.bind(self.get_map_of_stations, vartype=self.param.parameter_type))
        col2 = pn.Column(pn.bind(self.show_inventory,
                         index=self.station_select.param.index))
        return pn.Row(col1, col2)

#!conda install -y -c conda-forge jupyter_bokeh
if __name__ == '__main__':
    # using argparse to get the directory
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='directory with station inventory')
    args = parser.parse_args()
    # if no args exit with help message
    if args.dir is None:
        parser.print_help()
        exit(0)
    else:
        dir = args.dir
    #
    explorer = StationInventoryExplorer(dir)
    #
    explorer.create_maps_view().show(title='Station Inventory Explorer')
