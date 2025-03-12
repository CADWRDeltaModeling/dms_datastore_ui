# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
import holoviews as hv
from holoviews import opts
import panel as pn

hv.extension("bokeh")

all = ["plot_missing_data"]


def generate_sample_data():
    """
    Generate sample time series data with:
      - 12 series over a 2-year period at 15-minute intervals.
      - 20 random gaps (length 1 to 200 intervals) per series.
      - In 6 randomly chosen series, a gap at the end (up to 2 days missing).
      - In the first series, a gap covering the first year.
    """
    freq = "15min"
    start = pd.Timestamp("2010-01-01")
    end = pd.Timestamp("2012-01-01")
    index = pd.date_range(start, end, freq=freq)
    n = len(index)
    cols = [f"Series{i+1:02d}" for i in range(12)]

    # Create DataFrame with random data
    data = np.random.randn(n, 12)
    df = pd.DataFrame(data, index=index, columns=cols)

    # For reproducibility of gaps:
    rng = np.random.default_rng(seed=0)

    # Introduce 20 large and 20 small scattered gaps per series (randomly chosen start and gap length)
    for col in df.columns:
        for _ in range(20):
            start_idx = rng.integers(0, n - 1)
            gap_length = rng.integers(1, 201)  # gap length between 1 and 200 intervals
            end_idx = min(start_idx + gap_length, n)
            df.loc[df.index[start_idx:end_idx], col] = np.nan
            start_idx = rng.integers(0, n - 1)
            gap_length = rng.integers(1, 5)  # gap length between 1 and 5 intervals
            end_idx = min(start_idx + gap_length, n)
            df.loc[df.index[start_idx:end_idx], col] = np.nan

    # For 6 randomly chosen series, remove a gap at the end (up to 2 days missing)
    gap_end_candidates = rng.choice(df.columns, size=6, replace=False)
    intervals_per_day = int((24 * 60) / 15)
    for col in gap_end_candidates:
        gap_length = rng.integers(1, 2 * intervals_per_day + 1)
        df.loc[df.index[-gap_length:], col] = np.nan

    # For the first series, remove the first year of data
    df.loc[df.index < (start + pd.DateOffset(years=1)), df.columns[0]] = np.nan
    df.iloc[0 : (n - 400), 7] = 0.0
    df.iloc[:, 8] = 0.0
    return df[cols]


def plot_missing_data(df, ax, min_gap_duration, overall_start, overall_end):
    """
    Plot missing data onto the provided axis with a given minimum gap duration.
    """
    overall_start_num = mdates.date2num(overall_start)
    overall_end_num = mdates.date2num(overall_end)

    # Colors for the bars
    overall_color = "skyblue"
    gap_color = "orange"
    boundary_gap_color = "indianred"

    bar_height = 0.8  # thickness for each horizontal bar

    # Clear current content on ax
    ax.cla()

    # Prepare lists for y-ticks and labels with annotations.
    y_ticks = []
    y_labels = []

    # Loop over each series (each column in the DataFrame)
    for i, col in enumerate(df.columns):
        # Draw a light blue bar covering the entire time span for this series.
        ax.broken_barh(
            [(overall_start_num, overall_end_num - overall_start_num)],
            (i - bar_height / 2, bar_height),
            facecolors=overall_color,
            alpha=0.6,
        )

        # Extract the series and find the missing (NaN) segments.
        series = df[col]
        mask = series.isna()
        if mask.any():
            groups = (mask != mask.shift()).cumsum()
            for group_id, group in mask.groupby(groups):
                if group.iloc[0]:  # missing segment
                    gap_start = group.index[0]
                    gap_end = group.index[-1] + pd.Timedelta(minutes=15)

                    # Expand gap if too short
                    actual_gap = gap_end - gap_start
                    if actual_gap < min_gap_duration:
                        extra = min_gap_duration - actual_gap
                        gap_start_adj = gap_start - extra / 2
                        gap_end_adj = gap_end + extra / 2
                        gap_start = max(gap_start_adj, overall_start)
                        gap_end = min(gap_end_adj, overall_end)

                    # Use a distinct color if the gap touches either end.
                    if gap_start <= overall_start or gap_end >= overall_end:
                        current_gap_color = boundary_gap_color
                    else:
                        current_gap_color = gap_color

                    gap_start_num = mdates.date2num(gap_start)
                    gap_end_num = mdates.date2num(gap_end)
                    ax.broken_barh(
                        [(gap_start_num, gap_end_num - gap_start_num)],
                        (i - bar_height / 2, bar_height),
                        facecolors=current_gap_color,
                    )

        y_ticks.append(i)
        # Compute percentage of missing data for annotation.
        perc = series.isna().mean() * 100
        if perc == 0:
            label = f"{col} (0%)"
        elif perc > 0 and perc < 0.01:
            label = f"{col} (<0.01%)"
        else:
            label = f"{col} ({perc:.2f}%)"
        y_labels.append(label)

    # Format axes
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.xaxis_date()
    ax.set_xlabel("Time")
    ax.set_title(f"Missing Data Visualization (Min gap = {min_gap_duration})")
    ax.figure.autofmt_xdate()
    ax.figure.canvas.draw_idle()


def interactive_gap_plot(df):
    """
    Create an interactive plot that updates the minimum gap duration based on the current x-axis view.
    The mapping used here is:
       - >=20 years view: min gap = 1 day
       - >=10 years view: min gap = 12 hours
       - Otherwise:      min gap = 1 hour
    """
    # Overall full time range (fixed)
    overall_start = df.index[0]
    overall_end = df.index[-1] + pd.Timedelta(minutes=15)

    # Create figure and initial plot with a default min_gap_duration.
    default_min_gap = timedelta(hours=20)  # starting default
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_missing_data(df, ax, default_min_gap, overall_start, overall_end)

    def on_xlim_change(event_ax):
        print("xlim changed")
        # Determine current view duration
        xlim = event_ax.get_xlim()
        dt0 = mdates.num2date(xlim[0])
        dt1 = mdates.num2date(xlim[1])
        view_duration = dt1 - dt0
        years_view = view_duration.total_seconds() / (365.25 * 24 * 3600)

        # Adjust min_gap_duration based on view span
        if years_view >= 20:
            new_min_gap = timedelta(days=1)
        elif years_view >= 10:
            new_min_gap = timedelta(hours=12)
        else:
            new_min_gap = timedelta(hours=1)

        # Redraw the missing data visualization with the new min_gap_duration.
        plot_missing_data(df, ax, new_min_gap, overall_start, overall_end)

    # Connect the x-axis limits change event to our callback.
    ax.callbacks.connect("xlim_changed", on_xlim_change)

    plt.tight_layout()
    plt.show()


def plot_missing_data_hv(df, min_gap_duration, overall_start, overall_end):
    """
    Plot missing data using HoloViews with a given minimum gap duration.
    """
    # Use pandas timestamps directly instead of matplotlib date numbers

    # Colors for the bars
    overall_color = "skyblue"
    gap_color = "orange"
    boundary_gap_color = "indianred"

    # Create separate lists for background bars and gap bars
    background_bars = []
    gap_bars = []
    y_labels = []

    # Loop over each series (each column in the DataFrame)
    for i, col in enumerate(df.columns):
        # Draw a light blue bar covering the entire time span for this series.
        background_bars.append(
            hv.Rectangles(
                [(overall_start, i - 0.4, overall_end, i + 0.4)],
                ["x0", "y0", "x1", "y1"],
            ).opts(color=overall_color, alpha=0.6, line_color=None)
        )

        # Extract the series and find the missing (NaN) segments.
        series = df[col]
        mask = series.isna()
        if mask.any():
            groups = (mask != mask.shift()).cumsum()
            for group_id, group in mask.groupby(groups):
                if group.iloc[0]:  # missing segment
                    gap_start = group.index[0]
                    gap_end = group.index[-1] + pd.Timedelta(minutes=15)

                    # Expand gap if too short
                    actual_gap = gap_end - gap_start
                    if actual_gap < min_gap_duration:
                        extra = min_gap_duration - actual_gap
                        gap_start_adj = gap_start - extra / 2
                        gap_end_adj = gap_end + extra / 2
                        gap_start = max(gap_start_adj, overall_start)
                        gap_end = min(gap_end_adj, overall_end)

                    # Use a distinct color if the gap touches either end.
                    if gap_start <= overall_start or gap_end >= overall_end:
                        current_gap_color = boundary_gap_color
                    else:
                        current_gap_color = gap_color

                    # Calculate the duration of the gap for hover text
                    gap_duration = gap_end - gap_start
                    hours = gap_duration.total_seconds() / 3600

                    # Create the rectangle with all data including hover info
                    # Include all data from the start
                    data = {
                        "x0": [gap_start],
                        "y0": [i - 0.4],
                        "x1": [gap_end],
                        "y1": [i + 0.4],
                        "series": [col],
                        "duration_hours": [hours],
                    }

                    gap_rect = hv.Rectangles(
                        data,
                        kdims=["x0", "y0", "x1", "y1"],
                        vdims=["series", "duration_hours"],
                    )

                    # Add to gap bars with styling
                    gap_bars.append(
                        gap_rect.opts(color=current_gap_color, line_color=None)
                    )

        # Compute percentage of missing data for annotation.
        perc = series.isna().mean() * 100
        if perc == 0:
            label = f"{col} (0%)"
        elif perc > 0 and perc < 0.01:
            label = f"{col} (<0.01%)"
        else:
            label = f"{col} ({perc:.2f}%)"
        y_labels.append(label)

    # Combine background bars (no hover) and gap bars (with hover)
    background_overlay = hv.Overlay(background_bars).opts(
        opts.Rectangles(height=400, width=800, tools=[], line_color=None)
    )

    gap_overlay = hv.Overlay(gap_bars).opts(
        opts.Rectangles(
            tools=["hover"],
            hover_tooltips=[
                ("Series", "@series"),
                ("Start", "@x0{%F %H:%M}"),
                ("End", "@x1{%F %H:%M}"),
                ("Duration", "@duration_hours{0.2f} hours"),
            ],
            hover_formatters={"@x0": "datetime", "@x1": "datetime"},
            line_color=None,
        )
    )

    # Combine the overlays and set common properties
    combined = (background_overlay * gap_overlay).opts(
        opts.Overlay(
            height=400,
            width=800,
            xlabel="Time",
            ylabel="Series",
            title=f"Missing Data Visualization (Min gap = {min_gap_duration})",
            xrotation=45,
        )
    )

    # Set y-axis labels
    combined = combined.opts(opts.Rectangles(yticks=list(enumerate(y_labels))))

    return combined


def interactive_gap_plot_hv(df):
    """
    Create an interactive plot using HoloViews that updates the minimum gap duration based on the current x-axis view.
    """
    # Overall full time range (fixed)
    overall_start = df.index[0]
    overall_end = df.index[-1] + pd.Timedelta(minutes=15)

    # Create initial plot with a default min_gap_duration.
    default_min_gap = timedelta(hours=20)  # starting default
    initial_plot = plot_missing_data_hv(df, default_min_gap, overall_start, overall_end)

    # Create a callback that will update the plot when the x-range changes
    def update_plot(x_range=None):
        if x_range is None:
            # Initial plot case
            return initial_plot

        start, end = x_range

        # Convert numerical x_range to timestamps if needed
        if isinstance(start, (int, float)):
            start = pd.Timestamp(mdates.num2date(start))
        if isinstance(end, (int, float)):
            end = pd.Timestamp(mdates.num2date(end))

        # Calculate view span
        view_duration = end - start

        # Convert numpy.timedelta64 to Python timedelta for total_seconds() method
        if isinstance(view_duration, np.timedelta64):
            view_duration = pd.Timedelta(view_duration).to_pytimedelta()

        years_view = view_duration.total_seconds() / (365.25 * 24 * 3600)

        # Adjust min_gap_duration based on view span
        if years_view >= 20:
            new_min_gap = timedelta(days=1)
        elif years_view >= 10:
            new_min_gap = timedelta(hours=12)
        else:
            new_min_gap = timedelta(hours=1)

        return plot_missing_data_hv(df, new_min_gap, overall_start, overall_end)

    # Create a dynamic map with a RangeX stream
    range_stream = hv.streams.RangeX()
    dmap = hv.DynamicMap(update_plot, streams=[range_stream])

    # Return the plot wrapped in a Panel
    return pn.pane.HoloViews(dmap)


def plot_missing_data_hv_selectable(df, min_gap_duration, overall_start, overall_end):
    """
    Plot missing data using HoloViews with a given minimum gap duration.
    Makes gaps selectable with tap tool.
    """
    # Use pandas timestamps directly instead of matplotlib date numbers

    # Colors for the bars
    overall_color = "skyblue"
    gap_color = "orange"
    boundary_gap_color = "indianred"

    # Create separate lists for background bars and gap bars
    background_bars = []
    gap_bars = []
    y_labels = []

    # Loop over each series (each column in the DataFrame)
    for i, col in enumerate(df.columns):
        # Draw a light blue bar covering the entire time span for this series.
        background_bars.append(
            hv.Rectangles(
                [(overall_start, i - 0.4, overall_end, i + 0.4)],
                ["x0", "y0", "x1", "y1"],
            ).opts(color=overall_color, alpha=0.6, line_color=None)
        )

        # Extract the series and find the missing (NaN) segments.
        series = df[col]
        mask = series.isna()
        if mask.any():
            groups = (mask != mask.shift()).cumsum()
            for group_id, group in mask.groupby(groups):
                if group.iloc[0]:  # missing segment
                    gap_start = group.index[0]
                    gap_end = group.index[-1] + pd.Timedelta(minutes=15)

                    # Expand gap if too short
                    actual_gap = gap_end - gap_start
                    if actual_gap < min_gap_duration:
                        extra = min_gap_duration - actual_gap
                        gap_start_adj = gap_start - extra / 2
                        gap_end_adj = gap_end + extra / 2
                        gap_start = max(gap_start_adj, overall_start)
                        gap_end = min(gap_end_adj, overall_end)

                    # Use a distinct color if the gap touches either end.
                    if gap_start <= overall_start or gap_end >= overall_end:
                        current_gap_color = boundary_gap_color
                    else:
                        current_gap_color = gap_color

                    # Calculate the duration of the gap for hover text
                    gap_duration = gap_end - gap_start
                    hours = gap_duration.total_seconds() / 3600

                    # Create the rectangle with all data including hover info
                    data = {
                        "x0": [gap_start],
                        "y0": [i - 0.4],
                        "x1": [gap_end],
                        "y1": [i + 0.4],
                        "series": [col],
                        "duration_hours": [hours],
                    }

                    gap_rect = hv.Rectangles(
                        data,
                        kdims=["x0", "y0", "x1", "y1"],
                        vdims=["series", "duration_hours"],
                    )

                    # Add to gap bars with styling
                    gap_bars.append(
                        gap_rect.opts(color=current_gap_color, line_color=None)
                    )

        # Compute percentage of missing data for annotation.
        perc = series.isna().mean() * 100
        if perc == 0:
            label = f"{col} (0%)"
        elif perc > 0 and perc < 0.01:
            label = f"{col} (<0.01%)"
        else:
            label = f"{col} ({perc:.2f}%)"
        y_labels.append(label)

    gap_overlay = hv.Overlay(gap_bars).opts(
        opts.Rectangles(
            tools=["hover", "tap"],  # Explicitly add tap tool here
            active_tools=["tap"],  # Make tap the active tool
            hover_tooltips=[
                ("Series", "@series"),
                ("Start", "@x0{%F %H:%M}"),
                ("End", "@x1{%F %H:%M}"),
                ("Duration", "@duration_hours{0.2f} hours"),
            ],
            hover_formatters={"@x0": "datetime", "@x1": "datetime"},
            line_color=None,
            selection_color="red",  # Highlight selected rectangle
            nonselection_alpha=0.6,  # Make non-selected rectangles slightly transparent
        )
    )

    return gap_overlay


def interactive_gap_plot_hv_with_selection(df):
    """
    Create an interactive plot using HoloViews that:
    1. Updates the minimum gap duration based on the current x-axis view
    2. Shows a time series plot for the selected gap with 5 days of context on either side
    """
    # Overall full time range (fixed)
    overall_start = df.index[0]
    overall_end = df.index[-1] + pd.Timedelta(minutes=15)

    # Create initial plot with a default min_gap_duration.
    default_min_gap = timedelta(hours=20)  # starting default

    # Create streams for range and tap selection
    range_stream = hv.streams.RangeX()
    tap_stream = hv.streams.Selection1D()  # Create without source initially

    # Store the last selected index to preserve selection
    last_selected_index = [None]
    last_selected_data = [None]  # Store rectangle data for persistence

    # Create initial plot and properly connect the source
    initial_gap_plot = plot_missing_data_hv_selectable(
        df, default_min_gap, overall_start, overall_end
    )
    gap_overlay = initial_gap_plot.get(1)  # Get the gap overlay
    tap_stream.source = gap_overlay  # Connect to the gap overlay

    # Create a background plot to show the full time range
    def update_background_plot(df, x_range=None):
        overall_color = "skyblue"
        # Create separate lists for background bars and gap bars
        background_bars = []
        gap_bars = []
        y_labels = []

        # Loop over each series (each column in the DataFrame)
        for i, col in enumerate(df.columns):
            # Draw a light blue bar covering the entire time span for this series.
            background_bars.append(
                hv.Rectangles(
                    [(overall_start, i - 0.4, overall_end, i + 0.4)],
                    ["x0", "y0", "x1", "y1"],
                ).opts(color=overall_color, alpha=0.6, line_color=None)
            )
        # Combine background bars (no hover) and gap bars (with hover and tap)
        background_overlay = hv.Overlay(background_bars).opts(
            opts.Rectangles(height=400, width=800, tools=[], line_color=None)
        )
        return background_overlay

    def get_gap_duration(x_range=None):
        start, end = x_range

        # Convert numerical x_range to timestamps if needed
        if isinstance(start, (int, float)):
            start = pd.Timestamp(mdates.num2date(start))
        if isinstance(end, (int, float)):
            end = pd.Timestamp(mdates.num2date(end))

        # Calculate view span
        view_duration = end - start

        # Convert numpy.timedelta64 to Python timedelta for total_seconds() method
        if isinstance(view_duration, np.timedelta64):
            view_duration = pd.Timedelta(view_duration).to_pytimedelta()

        years_view = view_duration.total_seconds() / (365.25 * 24 * 3600)

        # Adjust min_gap_duration based on view span
        if years_view >= 20:
            new_min_gap = timedelta(days=1)
        elif years_view >= 10:
            new_min_gap = timedelta(hours=12)
        else:
            new_min_gap = timedelta(hours=1)
        return new_min_gap

    # Create a callback that will update the plot when the x-range changes
    def update_gap_plot(x_range=None):
        if x_range is None:
            # Initial plot case
            return initial_gap_plot

        new_min_gap = get_gap_duration(x_range)
        # Store current selection before updating plot
        if hasattr(tap_stream, "index") and tap_stream.index:
            last_selected_index[0] = tap_stream.index[0]
            # Also store the data for this selection
            try:
                element = tap_stream.source
                if element and isinstance(element, hv.Overlay):
                    rect_elements = [
                        el for el in element if isinstance(el, hv.Rectangles)
                    ]
                    if rect_elements and last_selected_index[0] is not None:
                        idx = last_selected_index[0]
                        rect_element = rect_elements[0]
                        data_dict = rect_element.data
                        if isinstance(data_dict, pd.DataFrame):
                            last_selected_data[0] = dict(data_dict.iloc[idx])
                        else:
                            # Get all data for this index
                            last_selected_data[0] = {
                                k: v[idx] for k, v in data_dict.items()
                            }
            except:
                pass

        updated_plot = plot_missing_data_hv_selectable(
            df, new_min_gap, overall_start, overall_end
        )

        # # Re-trigger the time series update with the stored selection
        # if last_selected_index[0] is not None:
        #     # This will make the Selection1D stream emit again with the last index
        #     tap_stream.event(index=[last_selected_index[0]])

        return updated_plot

    # Function to update the time series plot when a gap is selected
    def update_time_series(index):
        # Save the current selection for persistence
        if index is not None and len(index) > 0:
            last_selected_index[0] = index[0]

            # Get the currently selected rectangle data
            element = tap_stream.source
            if element and isinstance(element, hv.Overlay):
                rect_elements = [el for el in element if isinstance(el, hv.Rectangles)]
                if rect_elements:
                    rect_element = rect_elements[0]
                    data_dict = rect_element.data

                    # Get data for the selected index
                    idx = index[0]
                    if isinstance(data_dict, pd.DataFrame):
                        rect_data = dict(data_dict.iloc[idx])
                    else:
                        rect_data = {k: v[idx] for k, v in data_dict.items()}

                    # Store this data for persistence
                    last_selected_data[0] = rect_data

                    # Extract relevant data for plotting
                    series_name = rect_data["series"]
                    start_time = rect_data["x0"]
                    end_time = rect_data["x1"]

                    # Get the data for this series
                    series_data = df[series_name]

                    # Get 5 days before and after the gap
                    context_period = pd.Timedelta(days=5)
                    before_start = start_time - context_period
                    after_end = end_time + context_period

                    # Filter the series to the context period
                    context_data = series_data.loc[before_start:after_end]

                    # Create the curve data and plot
                    curve_data = [
                        (t, v)
                        for t, v in zip(context_data.index, context_data.values)
                        if pd.notna(v)
                    ]
                    curve = hv.Curve(
                        curve_data, "Time", "Value", label=series_name
                    ).opts(
                        height=250,
                        width=800,
                        title=f"Time Series: {series_name} (5 days before and after gap)",
                        ylabel="Value",
                        tools=["hover"],
                        legend_position="top_left",
                    )

                    # Add shaded region
                    gap_region = hv.VSpan(start_time, end_time).opts(
                        color="red", alpha=0.2
                    )
                    return curve * gap_region

        # For cases with no current selection:
        elif last_selected_data[0] is not None:
            # Create time series using the stored data
            series_name = last_selected_data[0]["series"]
            start_time = last_selected_data[0]["x0"]
            end_time = last_selected_data[0]["x1"]

            # Continue with the existing code to create plot using these values
            # Get the data for this series
            series_data = df[series_name]

            # Get 5 days before and after the gap
            context_period = pd.Timedelta(days=5)
            before_start = start_time - context_period
            after_end = end_time + context_period

            # Filter the series to the context period
            context_data = series_data.loc[before_start:after_end]

            # Create the curve data and plot
            curve_data = [
                (t, v)
                for t, v in zip(context_data.index, context_data.values)
                if pd.notna(v)
            ]
            curve = hv.Curve(curve_data, "Time", "Value", label=series_name).opts(
                height=250,
                width=800,
                title=f"Time Series: {series_name} (5 days before and after gap)",
                ylabel="Value",
                tools=["hover"],
                legend_position="top_left",
            )

            # Add shaded region
            gap_region = hv.VSpan(start_time, end_time).opts(color="red", alpha=0.2)
            return curve * gap_region
        else:
            # No selection and no stored data
            return hv.Curve([], "Time", "Value").opts(
                height=250, width=800, title="No gap selected", ylabel="Value"
            )

    # Create dynamic maps for the gap plot and time series plot
    gap_dmap = hv.DynamicMap(update_gap_plot, streams=[range_stream])
    background_dmap = hv.DynamicMap(update_background_plot(df), streams=[range_stream])
    tap_stream.source = gap_dmap

    ts_dmap = hv.DynamicMap(update_time_series, streams=[tap_stream])

    # Combine the overlays and set common properties
    combined = (background_dmap * gap_dmap).opts(
        opts.Overlay(
            height=400,
            width=800,
            xlabel="Time",
            ylabel="Series",
            # title=f"Missing Data Visualization (Min gap = {min_gap_duration})",
            xrotation=45,
        )
    )

    # Set y-axis labels
    # combined = combined.opts(opts.Rectangles(yticks=list(enumerate(y_labels))))

    # Return both plots in a layout
    return pn.Column(
        pn.pane.Markdown(
            "## Missing Data Visualization\n*Select a gap (orange/red bar) to see time series data*"
        ),
        pn.pane.HoloViews(combined),
        pn.pane.HoloViews(ts_dmap),
    )


# %%
def gap_visualizer(df, column_name="value"):
    """
    Using curve and spikes display the missing data.
    """
    missing_df = df[df[column_name].isna()][[column_name]]
    # Create a curve plot
    curve = hv.Curve(df, "index", column_name, label=column_name).opts(
        height=400,
        width=800,
        title="Time Series Data",
        xlabel="Time",
        ylabel="column_name",
    )
    # Create a spikes plot
    spikes = hv.Spikes(missing_df, "index", [], label="Missing").opts(
        height=400,
        width=800,
        title="Missing Data Visualization",
        xlabel="Time",
        ylabel="Series",
    )
    return (
        (curve + spikes)
        .opts(
            opts.Curve(
                height=200,
                xaxis=None,
                line_width=1.50,
                color="red",
                tools=["hover"],
            ),
            opts.Spikes(height=150, yaxis=None, line_width=0.5, color="grey"),
        )
        .cols(1)
    )


# %%

# --- Main ---
if __name__ == "__main__":
    try:
        df_sample = generate_sample_data()
        # Use the interactive plot with selection
        interactive_plot = interactive_gap_plot_hv_with_selection(df_sample)

        # Create a panel app with error handling
        app = pn.Column(
            pn.pane.Markdown("# Gap Visualization with Selection"), interactive_plot
        )

        # Serve the app - use show() for notebooks, serve() for command line
        # app.show(title="Gap Visualization with Selection")

    except Exception as e:
        import traceback

        print(f"Error occurred: {e}")
        traceback.print_exc()

# %%
