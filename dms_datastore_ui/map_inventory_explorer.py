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
pn.extension(notifications=True)
import param
#
from vtools.functions.filter import godin, cosine_lanczos

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
from bokeh.core.enums import MarkerType
#print(list(MarkerType))
#['asterisk', 'circle', 'circle_cross', 'circle_dot', 'circle_x', 'circle_y', 'cross', 'dash',
# 'diamond', 'diamond_cross', 'diamond_dot', 'dot', 'hex', 'hex_dot', 'inverted_triangle', 'plus',
# 'square', 'square_cross', 'square_dot', 'square_pin', 'square_x', 'star', 'star_dot',
# 'triangle', 'triangle_dot', 'triangle_pin', 'x', 'y']
param_to_marker_map = {'elev': 'square', 'predictions': 'square_x', 'turbidity':'diamond',
                       'flow': 'circle', 'velocity': 'circle_dot', 'temp': 'cross',
                       'do': 'asterisk', 'ec': 'triangle', 'ssc': 'diamond',
                       'ph': 'plus', 'salinity': 'inverted_triangle', 'cla': 'dot', 'fdom': 'hex'}
class StationInventoryExplorer(param.Parameterized):
    '''
    Show station inventory on map and select to display data available
    Furthermore select the data rows and click on button to display plots for selected rows
    '''
    time_window = param.CalendarDateRange(default=(datetime.now()- timedelta(days=10), datetime.now()), doc="Time window for data. Default is last 10 days")
    repo_level = param.ListSelector(objects=['formatted_1yr', 'screened'], default=['formatted_1yr'],
                                doc='repository level (sub directory) under which data is found. You can select multiple repo levels (ctrl+click)')
    parameter_type = param.ListSelector(objects=['all'],
                                    default=['all'],
                                    doc='parameter type of data, e.g. flow, elev, temp, etc. You can select multiple parameters (ctrl+click) or all'
                                    )
    apply_filter = param.Boolean(default=False, doc='Apply tidal filter to data')
    filter_type = param.Selector(objects=['cosine_lanczos', 'godin'], default='cosine_lanczos', doc='Filter type is cosine lanczos with a 40 hour cutoff or godin')
    map_color_category = param.Selector(objects=['param', 'agency'  ], default='param', doc='Color by parameter or agency')
    use_symbols_for_params = param.Boolean(default=False, doc='Use symbols for parameters. If not selected, all parameters will be shown as circles')
    search_text = param.String(default='', doc='Search text to filter stations')
    show_legend = param.Boolean(default=True, doc='Show legend')
    legend_position = param.Selector(objects=['top_right', 'top_left', 'bottom_right', 'bottom_left'], default='top_right', doc='Legend position')
    sensible_range_yaxis = param.Boolean(default=False, doc='Sensible range (1st and 99th percentile) or auto range for y axis')

    def __init__(self, dir, **kwargs):
        super().__init__(**kwargs)
        self.dir = dir
        self.inventory_file, mtime = find_lastest_fname(
            'inventory*.csv', self.dir)
        self.df_dataset_inventory = pd.read_csv(
            os.path.join(self.dir, self.inventory_file))
        # replace nan with empty string for column subloc
        self.df_dataset_inventory['subloc'] = self.df_dataset_inventory['subloc'].fillna('')
        self.unique_params = self.df_dataset_inventory['param'].unique()
        self.param.parameter_type.objects = ['all'] +  list(self.unique_params)
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
                                              ).opts(size=6, color=dim(self.map_color_category), cmap='Category10',
                                                     #marker=dim('param').categorize(param_to_marker_map),
                                                     tools=[hover], height=800)
        self.map_station_inventory = self.map_station_inventory.opts(opts.Points(tools=['tap', hover, 'lasso_select', 'box_select'],
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

    def get_param_to_marker_map(self):
        if self.use_symbols_for_params:
            return param_to_marker_map
        else:
            return {p: 'circle' for p in self.unique_params}

    @param.depends('search_text', watch=True)
    def do_search(self):
        # Create a boolean mask to select rows with matching text
        mask = self.current_station_inventory.apply(lambda row: row.astype(str).str.contains(self.search_text, case=False).any(), axis=1)
        # Use the boolean mask to select the matching rows
        index = self.current_station_inventory.index[mask]
        self.station_select.event(index = list(index)) # this should trigger show_inventory

    def save_dataframe(self, event):
        df = self.display_table.value.iloc[self.display_table.selection]
        df = df.merge(self.df_dataset_inventory)
        for i, r in df.iterrows():
            dfdata = self.get_data_for(r)
            param = r['param']
            unit = r['unit']
            station_id = r['station_id']
            agency = r['agency']
            dfdata.to_csv(f'saved_{agency}_{station_id}_{param}.csv')

    def _append_to_title_map(self, title_map, r, repo_level):
        value = title_map[r['unit']]
        if repo_level not in value[0]:
            value[0] += f',{repo_level}'
        if r['station_id'] not in value[2]:
            value[2] += f',{r["station_id"]}'
        if r['agency'] not in value[3]:
            value[3] += f',{r["agency"]}'
        title_map[r['unit']] = value

    def _create_title(self, v):
        title = f'{v[1]} @ {v[2]} ({v[3]}::{v[0]})'
        return title

    def _calculate_range(self, current_range, df, factor=0.1):
        if df.empty:
            return current_range
        else:
            new_range = df.iloc[:,1].quantile([0.05, 0.995]).values
            scaleval = new_range[1]-new_range[0]
            new_range = [new_range[0]-scaleval*factor, new_range[1]+scaleval*factor]
        if current_range is not None:
            new_range = [min(current_range[0], new_range[0]), max(current_range[1], new_range[1])]
        return new_range

    def create_plots(self, event):
        #df = self.display_table.selected_dataframe # buggy
        df = self.display_table.value.iloc[self.display_table.selection]
        df = df.merge(self.df_dataset_inventory)
        try:
            layout_map = {}
            title_map = {}
            range_map = {}
            for i, r in df.iterrows():
                for repo_level in self.repo_level:
                    crv = self.create_curve(r, repo_level)
                    unit = r['unit']
                    if unit not in layout_map:
                        layout_map[unit] = []
                        title_map[unit] = [repo_level, r['param'], r['station_id'], r['agency'], r['subloc']]
                        range_map[unit] = None
                    layout_map[unit].append(crv)
                    if self.sensible_range_yaxis:
                        range_map[unit] = self._calculate_range(range_map[unit], crv.data)
                    self._append_to_title_map(title_map, r, repo_level)
            if len(layout_map) == 0:
                return hv.Div('<h3>Select rows from table and click on button</h3>')
            else:
                return hv.Layout([hv.Overlay(layout_map[k]).opts(show_legend=self.show_legend, legend_position=self.legend_position,
                                                                 ylim=tuple(range_map[k]) if range_map[k] is not None else (None, None),
                                                                 title=self._create_title(title_map[k])) for k in layout_map]).cols(1).opts(axiswise=True)
        except Exception as e:
            stackmsg = full_stack()
            print(stackmsg)
            pn.state.notifications.error(f'Error while fetching data for {e}')
            return hv.Div(f'<h3> Exception while fetching data </h3> <pre>{e}</pre>')

    def create_curve(self, r, repo_level):
        filename = r['filename']
        param = r['param']
        unit = r['unit']
        station_id = r['station_id']
        subloc = r["subloc"] if len(r['subloc']) == 0 else f'/{r["subloc"]}'
        agency = r['agency']
        agency_id_dbase = r['agency_id_dbase']
        df = self.get_data_for(r, repo_level)
        if self.apply_filter:
            df = df.interpolate(limit_direction='both', limit=10)
            if self.filter_type == 'cosine_lanczos':
                if len(df) > 0:
                    df['value'] = cosine_lanczos(df['value'], '40H')
            else:
                if len(df) > 0:
                    df['value'] = godin(df['value'])
        crvlabel = f'{repo_level}/{station_id}{subloc}/{param}'
        crv = hv.Curve(df[['value']],label=crvlabel).redim(value=crvlabel)
        return crv.opts(xlabel='Time', ylabel=f'{param}({unit})', title=f'{repo_level}/{station_id}{subloc}::{agency}/{agency_id_dbase}', responsive=True, active_tools=['wheel_zoom'], tools=['hover'])

    def get_data_for(self, r, repo_level):
        filename = r['filename']
        try:
            df = get_station_data_for_filename(os.path.join(repo_level, filename), self.dir)
        except Exception as e:
            print(e)
            pn.state.notifications.error(f'Error while fetching data for {repo_level}/{filename}: {e}')
            df=pd.DataFrame(columns=['value'])
        df = df.loc[slice(*self.time_window), :]
        return df

    def cache(self, repo_level):
        # get unique filenames
        filenames = self.df_dataset_inventory['filename'].unique()
        print('Caching: ', len(filenames), ' files')
        for i, filename in enumerate(filenames):
            print(f'Caching {i} ::{filename}')
            try:
                get_station_data_for_filename(os.path.join(repo_level, filename), self.dir)
            except Exception as e:
                print(e)
                print('Skipping', filename, 'due to error')

    def update_plots(self, event):
        self.plot_panel.loading = True
        self.plot_panel.object = self.create_plots(event)
        self.plot_panel.loading = False

    def download_data(self):
        self.download_button.loading = True
        try:
            df = self.display_table.value.iloc[self.display_table.selection]
            df = df.merge(self.df_dataset_inventory)
            dflist = []
            for i, r in df.iterrows():
                for repo_level in self.repo_level:
                    dfdata = self.get_data_for(r, repo_level)
                    param = r['param']
                    unit = r['unit']
                    subloc = r['subloc']
                    station_id = r['station_id']
                    agency = r['agency']
                    agency_id_dbase = r['agency_id_dbase']
                    dfdata.columns = [f'{repo_level}/{station_id}/{subloc}/{agency}/{agency_id_dbase}/{param}/{unit}']
                    dflist.append(dfdata)
            dfdata = pd.concat(dflist, axis=1)
            sio = StringIO()
            dfdata.to_csv(sio)
            sio.seek(0)
            return sio
        finally:
            self.download_button.loading = False

    def update_data_table(self, dfs):
        # if attribute display_table is not set, create it
        if not hasattr(self, 'display_table'):
            column_width_map = {'index': '5%', 'station_id': '10%', 'subloc': '5%', 'lat': '8%', 'lon': '8%', 'name': '25%',
                                'min_year': '5%', 'max_year':'5%', 'agency': '5%', 'agency_id_dbase': '5%', 'param': '7%', 'unit': '8%'}
            self.display_table = pn.widgets.Tabulator(dfs, disabled=True, widths=column_width_map, show_index=False)
            self.plot_button = pn.widgets.Button(name="Plot", button_type="primary", icon='chart-line')
            self.plot_button.on_click(self.update_plots)
            self.plot_panel = pn.panel(hv.Div('<h3>Select rows from table and click on button</h3>'))
            # add a button to trigger the save function
            self.download_button = pn.widgets.FileDownload(label='Download', callback=self.download_data, filename='dms_data.csv',
                                                           button_type='primary', icon='file-download', embed=False)
            gspec = pn.GridSpec(sizing_mode='stretch_both')#,
            gspec[0,:3] = pn.Row(self.plot_button, self.download_button, sizing_mode='stretch_height')
            gspec[1:5,0:10] = self.display_table
            gspec[5:15,0:10] = self.plot_panel
            self.plots_panel = pn.Row(gspec) # fails with object of type 'GridSpec' has no len()
        else:
            self.display_table.value = dfs
        return self.plots_panel

    def get_map_of_stations(self, vartype, color_category, symbol_category):
        if len(vartype)==1 and vartype[0] == 'all':
            dfs = self.df_station_inventory
        else:
            dfs = self.df_station_inventory[self.df_station_inventory['param'].isin(vartype)]
        self.current_station_inventory = dfs
        self.map_station_inventory.data = self.current_station_inventory
        return self.tmap*self.map_station_inventory.opts(color=dim(color_category), marker=dim('param').categorize(self.get_param_to_marker_map()))

    def create_maps_view(self):
        control_widgets = pn.Param(self, widgets={"time_window": pn.widgets.DatetimeRangePicker})
        col1 = pn.Column(control_widgets, pn.bind(self.get_map_of_stations,
                                                  vartype=self.param.parameter_type,
                                                  color_category=self.param.map_color_category,
                                                  symbol_category=self.param.use_symbols_for_params), width=600)
        col2 = pn.Column(pn.bind(self.show_inventory, index=self.station_select.param.index))
        return pn.Row(col1, col2, sizing_mode='stretch_both')

    def create_view(self):
        control_widgets = pn.Row(
            pn.Column(
                    pn.Param(self.param.time_window, widgets={"time_window": {'widget_type': pn.widgets.DatetimeRangeInput, 'format': '%Y-%m-%d %H:%M'}}),
                    self.param.repo_level, self.param.parameter_type),
            pn.Column(self.param.apply_filter, self.param.filter_type,
                      self.param.show_legend, self.param.legend_position,
                      self.param.map_color_category, self.param.use_symbols_for_params,
                      self.param.sensible_range_yaxis,
                      self.param.search_text)
        )
        map_tooltip = pn.widgets.TooltipIcon(value='Map of stations. Click on a station to see data available in the table. See <a href="https://docs.bokeh.org/en/latest/docs/user_guide/interaction/tools.html">Bokeh Tools</a> for toolbar operation')
        map_display = pn.bind(self.get_map_of_stations,
                                                  vartype=self.param.parameter_type,
                                                  color_category=self.param.map_color_category,
                                                  symbol_category=self.param.use_symbols_for_params)
        sidebar_view = pn.Column(control_widgets, pn.Column(pn.Row('Station Map',map_tooltip), map_display))
        main_view = pn.Column(pn.bind(self.show_inventory, index=self.station_select.param.index))
        # Add disclaimer about data hosted here
        disclaimer_text = """
        ## Disclaimer

        The data here is not the original data as provided by the agencies. The original data should be obtained from the agencies.

        The data presented here is an aggregation of data from various sources. The various sources are listed in the inventory file as agency and agency_id_dbase.

        The data here has been modified and corrected as needed by the Delta Modeling Section for use in the Delta Modeling Section's models and analysis.
        """
        #
        template = pn.template.MaterialTemplate(title='DMS Datastore',sidebar=[sidebar_view],
                                                sidebar_width=650, header_color='blue', logo='https://sciencetracker.deltacouncil.ca.gov/themes/custom/basic/images/logos/DWR_Logo.png')
        template.modal.append(disclaimer_text)
        # Adding about button
        about_btn = pn.widgets.Button(name="About this Site", button_type="primary", icon="info-circle")
        def about_callback(event):
            template.open_modal()
        about_btn.on_click(about_callback)
        template.sidebar.append(about_btn)
        # Append a layout to the main area, to demonstrate the list-like API
        template.main.append(main_view)
        return template

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
        for repo_level in explorer.repo_level:
            print('Caching ', repo_level)
            explorer.cache(repo_level)
    else: # display ui
        explorer.create_view().show(title='Station Inventory Explorer')
