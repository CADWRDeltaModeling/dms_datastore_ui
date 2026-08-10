from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Callable, Dict, List, Optional
import logging


@dataclass
class PluginRegistration:
    """Container for plugin-provided dynamic apps and static directories."""

    name: str
    apps: Dict[str, Callable] = field(default_factory=dict)
    static_dirs: Dict[str, str] = field(default_factory=dict)
    description: str = ""


def _normalise_route(route: str) -> str:
    route = str(route).strip()
    if route == "/":
        return ""
    return route.lstrip("/").rstrip("/")


def _normalise_static_prefix(prefix: str) -> str:
    return str(prefix).strip().lstrip("/").rstrip("/")


def load_server_config(config_path: Optional[str]) -> dict:
    """Load server configuration from YAML.

    Returns an empty config when *config_path* is not provided.
    """
    if not config_path:
        return {}

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for --config support. Install with: pip install pyyaml"
        ) from exc

    cfg_path = Path(config_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Server config must be a mapping at top level: {cfg_path}")
    return loaded


def _to_registration(obj: object, default_name: str) -> PluginRegistration:
    """Accept either PluginRegistration or mapping-like plugin return values."""
    if isinstance(obj, PluginRegistration):
        return obj

    if not isinstance(obj, dict):
        raise TypeError(
            f"Plugin register() must return PluginRegistration or dict, got {type(obj)!r}"
        )

    return PluginRegistration(
        name=str(obj.get("name", default_name)),
        apps=dict(obj.get("apps", {}) or {}),
        static_dirs=dict(obj.get("static_dirs", {}) or {}),
        description=str(obj.get("description", "") or ""),
    )


def discover_entrypoint_plugins(
    plugin_config: Optional[dict],
    logger: logging.Logger,
    entrypoint_group: str = "dms_datastore_ui.server_plugins",
) -> List[PluginRegistration]:
    """Load plugin registrations from Python entry points."""
    plugin_config = plugin_config or {}
    if not isinstance(plugin_config, dict):
        logger.warning(
            "Ignoring invalid plugins config (expected mapping, got %s).",
            type(plugin_config).__name__,
        )
        plugin_config = {}

    eps = metadata.entry_points()
    if hasattr(eps, "select"):
        group_eps = list(eps.select(group=entrypoint_group))
    else:
        group_eps = list(eps.get(entrypoint_group, []))

    registrations: List[PluginRegistration] = []
    for ep in group_eps:
        try:
            register_fn = ep.load()
            cfg_for_plugin = plugin_config.get(ep.name, {})
            reg_obj = register_fn(cfg_for_plugin)
            reg = _to_registration(reg_obj, default_name=ep.name)
            registrations.append(reg)
            logger.info("Loaded server plugin '%s' from %s", ep.name, ep.value)
        except Exception:
            logger.exception("Failed to load server plugin '%s'", ep.name)

    return registrations


def merge_plugin_apps(
    base_apps: Dict[str, Callable],
    plugin_regs: List[PluginRegistration],
    logger: logging.Logger,
) -> Dict[str, Callable]:
    """Merge plugin routes into base routes.

    Base routes are protected. Plugin-plugin collisions warn and last wins.
    """
    merged: Dict[str, Callable] = dict(base_apps)

    for reg in plugin_regs:
        for route, factory in reg.apps.items():
            norm_route = _normalise_route(route)
            if norm_route in base_apps:
                logger.warning(
                    "Plugin '%s' attempted to override protected base route '/%s'; skipping.",
                    reg.name,
                    norm_route,
                )
                continue
            if norm_route in merged:
                logger.warning(
                    "Route collision on '/%s': plugin '%s' overrides previous plugin route.",
                    norm_route,
                    reg.name,
                )
            merged[norm_route] = factory

    return merged


def merge_static_dirs(
    config_static_dirs,
    plugin_regs: List[PluginRegistration],
    logger: logging.Logger,
) -> Dict[str, str]:
    """Merge static directory mappings from config and plugins.

    Config supports either:
    - mapping: {prefix: path}
    - list of mappings: [{prefix: reports, path: /data/reports}, ...]
    """
    merged: Dict[str, str] = {}

    def _maybe_add(prefix: str, directory: str, source: str) -> None:
        norm_prefix = _normalise_static_prefix(prefix)
        if not norm_prefix:
            logger.warning("Ignoring empty static prefix from %s", source)
            return
        directory_path = Path(directory)
        if not directory_path.exists() or not directory_path.is_dir():
            logger.warning(
                "Ignoring static dir '%s' for prefix '/%s' from %s (missing or not a directory)",
                directory,
                norm_prefix,
                source,
            )
            return
        if norm_prefix in merged:
            logger.warning(
                "Static prefix collision on '/%s': replacing previous mapping with %s",
                norm_prefix,
                source,
            )
        merged[norm_prefix] = str(directory_path)

    if isinstance(config_static_dirs, dict):
        for prefix, directory in config_static_dirs.items():
            _maybe_add(str(prefix), str(directory), source="server config")
    elif isinstance(config_static_dirs, list):
        for item in config_static_dirs:
            if not isinstance(item, dict):
                logger.warning("Ignoring invalid static_dirs item (expected mapping): %r", item)
                continue
            prefix = item.get("prefix")
            directory = item.get("path")
            if prefix is None or directory is None:
                logger.warning("Ignoring static_dirs item missing prefix/path: %r", item)
                continue
            _maybe_add(str(prefix), str(directory), source="server config")
    elif config_static_dirs is not None:
        logger.warning("Ignoring unsupported static_dirs type: %s", type(config_static_dirs).__name__)

    for reg in plugin_regs:
        for prefix, directory in reg.static_dirs.items():
            _maybe_add(str(prefix), str(directory), source=f"plugin '{reg.name}'")

    return merged


def startup_summary(
    apps: Dict[str, Callable],
    static_dirs: Dict[str, str],
    logger: logging.Logger,
) -> None:
    routes = []
    for route in apps:
        routes.append("/" if route == "" else f"/{route}")
    routes.sort()

    prefixes = sorted(f"/{k} -> {v}" for k, v in static_dirs.items())

    logger.info("Route summary: %s", ", ".join(routes) if routes else "<none>")
    logger.info("Static dirs: %s", ", ".join(prefixes) if prefixes else "<none>")
