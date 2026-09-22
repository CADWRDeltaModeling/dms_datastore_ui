from click.testing import CliRunner

from dms_datastore_ui import cli


def test_cli_help():
    result = CliRunner().invoke(cli.main, ["--help"])

    assert result.exit_code == 0
    assert "show-repo" in result.output
