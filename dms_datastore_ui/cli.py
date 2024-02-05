import click
from dms_datastore_ui import __version__

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    pass

@click.command(help="Print the version of dms_datastore_ui")
def version():
    print(__version__)

@click.command(help="Show the station inventory explorer")
@click.argument('repo_dir', type=click.Path(exists=True))
def show_repo(repo_dir):
    import dms_datastore_ui.map_inventory_explorer as mie
    explorer = mie.StationInventoryExplorer(repo_dir)
    #
    ui = explorer.create_view().servable(title=f"Station Inventory Explorer: {repo_dir}")
    ui.show()

@click.command(help="Show the data screener")
@click.argument('filepath', type=click.Path(exists=True))
def data_screener(filepath):
    from dms_datastore_ui.data_screener import DataScreener
    screener = DataScreener(filepath)
    screener.view().show()

@click.command(help="Show the flag editor")
@click.argument('filepath', type=click.Path(exists=True))
def flag_editor(filepath):
    from dms_datastore import read_ts, auto_screen
    from dms_datastore_ui.flag_editor import FlagEditor
    meta, df = read_ts.read_flagged(
        filepath, apply_flags=False, return_flags=True, return_meta=True
    )
    editor = FlagEditor(df)
    editor.view().show()



main.add_command(version)
main.add_command(show_repo)
main.add_command(data_screener)
main.add_command(flag_editor)

if __name__ == '__main__':
    import sys
    sys.exit(main())
