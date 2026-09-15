"""FastMCP server instance, lifespan, and tool/resource/prompt registration."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from simconnect_mcp.connection import SimConnectManager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Manage SimConnect connection lifecycle."""
    manager = SimConnectManager()
    logger.info("SimConnect MCP server starting")
    try:
        yield {"manager": manager}
    finally:
        manager.disconnect()
        logger.info("SimConnect MCP server stopped")


mcp = FastMCP(
    "SimConnect MCP",
    lifespan=lifespan,
)

# --- Register tools, resources, and prompts from domain modules ---

from simconnect_mcp.prompts.templates import register_prompts  # noqa: E402
from simconnect_mcp.resources.documentation import register_doc_resources  # noqa: E402
from simconnect_mcp.resources.state import register_state_resources  # noqa: E402
from simconnect_mcp.tools.aircraft import get_aircraft_snapshot  # noqa: E402
from simconnect_mcp.tools.connection_tools import (  # noqa: E402
    connect_to_sim,
    disconnect_from_sim,
    get_connection_status,
)
from simconnect_mcp.tools.events import (  # noqa: E402
    search_events,
    trigger_custom_event,
    trigger_event,
)
from simconnect_mcp.tools.facilities import (  # noqa: E402
    get_facility_info,
    get_nearby_airports,
)
from simconnect_mcp.tools.flight import (  # noqa: E402
    create_ai_object,
    load_flight,
    load_flight_plan,
    save_flight,
)
from simconnect_mcp.tools.hubhop import (  # noqa: E402
    list_hubhop_aircraft,
    search_hubhop,
)
from simconnect_mcp.tools.lvars import (  # noqa: E402
    browse_lvar_catalog,
    execute_calculator_code,
    get_lvar,
    list_lvars,
    search_lvars,
    set_lvar,
)
from simconnect_mcp.tools.pmdg import (  # noqa: E402
    get_pmdg_cdu,
    get_pmdg_var,
    send_pmdg_event,
)
from simconnect_mcp.tools.simvars import (  # noqa: E402
    get_object_simvars,
    get_simvar,
    get_simvar_bulk,
    list_simvar_categories,
    search_simvars,
    set_simvar,
    watch_simvar,
)
from simconnect_mcp.tools.utilities import (  # noqa: E402
    send_sim_text,
    set_aircraft_position,
)


def _register(fn, name: str, title: str, *, read_only: bool, idempotent: bool = False,
              destructive: bool | None = None) -> None:
    """Register one tool with explicit behaviour annotations.

    destructiveHint is only meaningful when readOnlyHint is false, so it
    defaults to the inverse of read_only.
    """
    mcp.tool(
        name=name,
        title=title,
        annotations=ToolAnnotations(
            title=title,
            readOnlyHint=read_only,
            destructiveHint=(not read_only) if destructive is None else destructive,
            idempotentHint=idempotent,
            # Every tool talks to a live external system: the simulator, or
            # (HubHop's two tools) its community preset API.
            openWorldHint=True,
        ),
    )(fn)


# --- Connection ---
_register(connect_to_sim, "msfs_connect", "Connect to MSFS",
          read_only=False, destructive=False, idempotent=True)
_register(disconnect_from_sim, "msfs_disconnect", "Disconnect from MSFS",
          read_only=False, destructive=False, idempotent=True)
_register(get_connection_status, "msfs_get_connection_status", "Get Connection Status",
          read_only=True, idempotent=True)

# --- SimVars ---
_register(get_simvar, "msfs_get_simvar", "Read SimVar", read_only=True, idempotent=True)
_register(set_simvar, "msfs_set_simvar", "Write SimVar", read_only=False, idempotent=True)
_register(get_simvar_bulk, "msfs_get_simvars_bulk", "Read Multiple SimVars",
          read_only=True, idempotent=True)
_register(get_object_simvars, "msfs_get_object_simvars",
          "Read SimVars From An Object", read_only=True, idempotent=True)
_register(search_simvars, "msfs_search_simvars", "Search SimVars",
          read_only=True, idempotent=True)
_register(list_simvar_categories, "msfs_list_simvar_categories", "List SimVar Categories",
          read_only=True, idempotent=True)
_register(watch_simvar, "msfs_watch_simvar", "Watch SimVar Over Time",
          read_only=True, idempotent=False)

# --- Events ---
_register(trigger_event, "msfs_trigger_event", "Trigger Event", read_only=False)
_register(search_events, "msfs_search_events", "Search Events",
          read_only=True, idempotent=True)
_register(trigger_custom_event, "msfs_trigger_custom_event", "Trigger Custom Event",
          read_only=False)

# --- L-vars ---
_register(get_lvar, "msfs_get_lvar", "Read L-Var", read_only=True, idempotent=True)
_register(set_lvar, "msfs_set_lvar", "Write L-Var", read_only=False, idempotent=True)
_register(list_lvars, "msfs_list_lvars", "List Aircraft L-Vars",
          read_only=True, idempotent=True)
_register(execute_calculator_code, "msfs_execute_calculator_code", "Execute RPN Code",
          read_only=False)
_register(search_lvars, "msfs_search_lvars", "Search L-Vars", read_only=True, idempotent=True)
_register(browse_lvar_catalog, "msfs_browse_lvar_catalog", "Browse L-Var Catalogs",
          read_only=True, idempotent=True)

# --- Aircraft ---
_register(get_aircraft_snapshot, "msfs_get_aircraft_snapshot", "Get Aircraft Snapshot",
          read_only=True, idempotent=True)

# --- Facilities ---
_register(get_nearby_airports, "msfs_get_nearby_airports", "Get Nearby Airports",
          read_only=True, idempotent=True)
_register(get_facility_info, "msfs_get_facility_info", "Get Facility Info",
          read_only=True, idempotent=True)

# --- Utilities ---
_register(send_sim_text, "msfs_send_sim_text", "Show Text In Sim",
          read_only=False, destructive=False)
_register(set_aircraft_position, "msfs_set_aircraft_position", "Reposition Aircraft",
          read_only=False, idempotent=True)

# --- PMDG ---
_register(get_pmdg_var, "msfs_get_pmdg_var", "Read PMDG Variable",
          read_only=True, idempotent=True)
_register(get_pmdg_cdu, "msfs_get_pmdg_cdu", "Read PMDG CDU Screen",
          read_only=True, idempotent=True)
_register(send_pmdg_event, "msfs_send_pmdg_event", "Send PMDG Event", read_only=False)

# --- HubHop ---
# Unlike every tool above, these two reach an HTTP API, not the simulator --
# see tools/hubhop.py's module docstring. Both work with MSFS closed, so
# neither is wired through @require_connection.
_register(search_hubhop, "msfs_search_hubhop", "Search HubHop Presets",
          read_only=True, idempotent=True)
_register(list_hubhop_aircraft, "msfs_list_hubhop_aircraft", "List HubHop Aircraft",
          read_only=True, idempotent=True)

# --- Flight and scenario ---
# The three idempotentHints below were decided together, since loading a
# file and saving one are not the same question and the first draft got two
# of them inconsistent with each other.
#
# Idempotent: loading the same .FLT twice leaves the sim in the state that
# file describes, both times -- the second call adds no effect the first
# did not already have. Identical in character to load_flight_plan below,
# which is why both now carry the hint; only one of them did before.
_register(load_flight, "msfs_load_flight", "Load Saved Flight",
          read_only=False, idempotent=True)
# NOT idempotent, despite reading like a write-once operation. With
# overwrite=False the second call refuses and changes nothing -- but with
# overwrite=True it captures the flight state as it is *now*, so the file
# it leaves behind differs from the first call's whenever the aircraft has
# moved in between. idempotentHint is a static annotation and cannot vary
# per argument, so it has to describe the tool's weakest case, not its
# best one.
#
# destructive=False is a separate question and does hold: overwrite=False
# is the default, so the tool refuses to replace an existing file unless
# the caller opts in (tools/flight.py) -- without that guard this would be
# a false claim.
_register(save_flight, "msfs_save_flight", "Save Current Flight",
          read_only=False, destructive=False)
# Idempotent for the same reason as load_flight. Unlike save_flight,
# load_flight_plan has no overwrite-style guard: it replaces whatever
# flight plan is currently active with no prompt and no way to opt out, so
# it keeps the read_only=False default (destructive=True), matching
# load_flight rather than copying save_flight's override.
_register(load_flight_plan, "msfs_load_flight_plan", "Load Flight Plan",
          read_only=False, idempotent=True)
# NOT idempotent, and unlike save_flight there is no reading under which it
# could be: each call spawns another AI object at the same position, so
# calling it twice leaves two.
_register(create_ai_object, "msfs_create_ai_object", "Create AI Object",
          read_only=False)

# Register resources and prompts
register_doc_resources(mcp)
register_state_resources(mcp)
register_prompts(mcp)


_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}


def configure_logging() -> None:
    """Send logs to stderr only.

    This server speaks JSON-RPC over stdio; anything written to stdout
    corrupts the protocol stream. Defaults to WARNING because the vendored
    MobiFlight bridge is chatty at INFO. Override with SIMCONNECT_MCP_LOG_LEVEL.

    The level name is matched against a fixed set rather than looked up on the
    logging module: a bare getattr would resolve any all-caps attribute, and
    logging.BASIC_FORMAT is a format string that makes setLevel raise and takes
    the whole server down at startup.
    """
    level_name = os.environ.get("SIMCONNECT_MCP_LOG_LEVEL", "WARNING").strip().upper()
    level = getattr(logging, level_name) if level_name in _LOG_LEVELS else logging.WARNING

    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(level)


def main() -> None:
    """Run the MCP server with stdio transport."""
    configure_logging()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
