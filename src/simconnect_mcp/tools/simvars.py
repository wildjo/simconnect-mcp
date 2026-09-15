"""SimVar tools — get, set, search, bulk read, watch."""

from __future__ import annotations

import asyncio
import time
from typing import Annotated

from pydantic import Field

from simconnect_mcp.connection import SimConnectManager
from simconnect_mcp.data.simvar_catalog import (
    load_catalog,
    lookup,
    resolve_unit,
    search_catalog,
    suggest_names,
)
from simconnect_mcp.simvar_access import (
    SimVarBatchTimeoutError,
    SimVarError,
    SimVarNotFoundError,
    SimVarNotSettableError,
    SimVarTimeoutError,
    UnitMismatchError,
)
from simconnect_mcp.tools import handle_simconnect_errors, require_connection
from simconnect_mcp.tools.formatting import (
    DEFAULT_LIMIT,
    ResponseFormat,
    build_search_result,
)
from simconnect_mcp.tools.models import (
    CategoryList,
    SearchResult,
    SimVarBulkResult,
    SimVarObjectResult,
    SimVarValue,
    SimVarWriteResult,
    ToolError,
    WatchResult,
    WatchSample,
    error_from,
)

# get_simvar_bulk's list is caller-supplied and otherwise uncapped; without a
# limit, a large enough list makes read_many hold _sim_lock for the whole
# batch's total timeout budget (see SimVarAccessor.read_many). 100 keeps a
# single call well within a couple of seconds even in the worst case.
MAX_BULK_VARIABLES = 100


def _disambiguate_not_found(name: str, unit: str | None) -> ToolError | None:
    """Tell a bad unit apart from a bad name.

    SimConnect raises the same NAME_UNRECOGNIZED for an unknown variable and
    for a known variable with an invalid unit (verified against a live sim),
    so the exception alone cannot distinguish them. The catalog can: if we
    know this variable and the caller supplied a unit, the name is fine and
    the unit is what was rejected.

    Returns a UNIT_MISMATCH envelope, or None to fall through to
    SIMVAR_NOT_FOUND. `unit` must be truthy -- with no caller unit we used
    the catalog's own, so a failure there is a genuine name problem.
    """
    if not unit or lookup(name) is None:
        return None
    return ToolError(
        error="UNIT_MISMATCH",
        message=f"SimConnect rejected unit '{unit}' for SimVar '{name}'.",
        suggestion=(
            f"'{name}' is measured in '{resolve_unit(name, None)}'. Omit the unit "
            "argument to use that, or pass a compatible SimConnect unit."
        ),
    )


def _simvar_error_envelope(e: SimVarError, name: str, unit: str | None) -> ToolError:
    """Diagnose a SimVarError the same way for every SimVar-reading tool.

    Before this, three call sites gave three different diagnoses for the
    identical underlying failure (a bad unit on an otherwise valid,
    known variable): get_simvar correctly reported UNIT_MISMATCH naming the
    real unit; watch_simvar reported the unrelated SIMVAR_NOT_READABLE with
    "check the name" -- wrong advice, since the name was fine; and
    get_simvar_bulk surfaced read_many's raw SimVarNotFoundError message
    ("SimConnect does not recognise SimVar ..."), which is actively
    misleading when the unit, not the name, was the problem. get_simvar,
    watch_simvar and get_simvar_bulk all route through this now, so one
    situation gets one diagnosis everywhere.

    The catch-all also now delegates to error_from (models.py) instead of
    keeping a second, separately-maintained "SIMVAR_ERROR" code: a bare
    SimVarError used to be diagnosed as SIMVAR_ERROR here but SIMCONNECT_ERROR
    by error_from's own fallback (reached whenever get_simvar/set_simvar let a
    bare SimVarError fall through to the decorator) -- the same situation,
    two different codes depending on which tool happened to see it.
    """
    if isinstance(e, SimVarNotFoundError):
        mismatch = _disambiguate_not_found(name, unit)
        if mismatch is not None:
            return mismatch
        return ToolError(
            error="SIMVAR_NOT_FOUND",
            message=f"SimConnect does not recognise SimVar '{name}'",
            suggestion="Use msfs_search_simvars to find the correct name.",
            suggestions=suggest_names(name) or None,
        )
    if isinstance(e, UnitMismatchError):
        return ToolError(
            error="UNIT_MISMATCH",
            message=str(e),
            suggestion=(
                f"Check the units for '{name}' with msfs_search_simvars, "
                "or omit the unit argument to use the catalog default."
            ),
        )
    # SimVarTimeoutError (and anything else, including a bare SimVarError)
    # falls through to error_from: its SIM_TIMEOUT mapping in models.py's
    # _ERROR_CODES is already word-for-word what a bespoke branch here would
    # return, so a separate branch would be pure duplication -- exactly the
    # kind of two-copies-of-one-fact drift risk this function exists to
    # remove for the bare-SimVarError case below.
    return error_from(e)


@handle_simconnect_errors
@require_connection(needs_accessor=True)
async def get_simvar(
    name: Annotated[
        str,
        Field(description="SimVar name, e.g. 'PLANE_ALTITUDE' or 'AIRSPEED_INDICATED'",
              min_length=1, max_length=128),
    ],
    unit: Annotated[
        str | None,
        Field(description="Unit to read in, e.g. 'feet', 'meters', 'knots'. "
                          "Defaults to the catalog unit for this variable."),
    ] = None,
    index: Annotated[
        int | None,
        Field(description="Index for indexed SimVars such as engine number. "
                          "Index 0 is valid.", ge=0, le=64),
    ] = None,
) -> SimVarValue | ToolError:
    """Read a SimVar value by name, in the requested unit.

    Returns the value together with the unit it was actually read in.
    Use msfs_search_simvars first if you are unsure of the exact name or units.
    """
    manager = SimConnectManager()
    resolved_unit = resolve_unit(name, unit)
    try:
        value = await manager.run_sync(
            lambda: manager.accessor.read(name, unit=unit, index=index)
        )
    except SimVarError as e:
        return _simvar_error_envelope(e, name, unit)
    return SimVarValue(name=name, value=value, unit=resolved_unit, index=index)


@handle_simconnect_errors
@require_connection(needs_accessor=True)
async def set_simvar(
    name: Annotated[str, Field(description="SimVar name; must be settable",
                               min_length=1, max_length=128)],
    value: Annotated[float, Field(description="Value to write")],
    unit: Annotated[
        str | None,
        Field(description="Unit the value is expressed in. Defaults to the catalog unit."),
    ] = None,
    index: Annotated[
        int | None, Field(description="Index for indexed SimVars. Index 0 is valid.",
                          ge=0, le=64)
    ] = None,
) -> SimVarWriteResult | ToolError:
    """Write a value to a settable SimVar.

    Fails with a specific error if the sim rejects the write, rather than
    reporting success. Check the 'settable' flag with msfs_search_simvars first.
    """
    manager = SimConnectManager()
    resolved_unit = resolve_unit(name, unit)
    try:
        # verify=True re-reads after the write. SimConnect does NOT raise for a
        # write to a read-only variable -- it silently ignores it -- so read-back
        # is the only way to tell the caller whether the value actually landed.
        verified = await manager.run_sync(
            lambda: manager.accessor.write(
                name, value, unit=unit, index=index, verify=True
            )
        )
    except SimVarNotSettableError as e:
        return ToolError(
            error="SIMVAR_NOT_SETTABLE",
            message=str(e),
            suggestion=(
                f"'{name}' appears to be read-only. Look for an event that changes it "
                "with msfs_search_events, or an aircraft-specific L-var with "
                "msfs_search_lvars. "
                "Note the catalog's 'settable' flag is unreliable, so a variable it "
                "marks read-only may still accept writes."
            ),
        )
    except SimVarError as e:
        return _simvar_error_envelope(e, name, unit)

    warning = None
    if verified is False:
        warning = (
            f"The write was sent but '{name}' did not change. SimConnect does not "
            "reject writes to read-only variables, so this usually means the "
            "variable is not settable. It can also mean the sim immediately "
            "overrode the value."
        )
    return SimVarWriteResult(
        name=name,
        value_set=value,
        unit=resolved_unit,
        index=index,
        verified=verified,
        warning=warning,
    )


# Maps read_many()'s per-entry error_type string back to the exception class,
# so a bulk entry's failure can be diagnosed through the same
# _simvar_error_envelope logic get_simvar uses. read_many only hands back the
# class *name* (it isolates failures into plain dicts, not exception
# instances), so this is the inverse of `type(e).__name__`.
_ERROR_TYPE_MAP: dict[str, type[SimVarError]] = {
    "SimVarNotFoundError": SimVarNotFoundError,
    "UnitMismatchError": UnitMismatchError,
    "SimVarTimeoutError": SimVarTimeoutError,
    "SimVarBatchTimeoutError": SimVarBatchTimeoutError,
    "SimVarNotSettableError": SimVarNotSettableError,
}


def diagnose_bulk_entries(
    results: dict[str, dict], caller_units: dict[str, str | None] | None = None
) -> tuple[int, int]:
    """Diagnose every failed entry of a read_many result, in place.

    Returns (ok_count, error_count).

    read_many isolates failures into plain dicts carrying a raw exception
    message and the exception's class *name*, so without this an entry
    reaches the agent as an undiagnosed string with no error code and no
    suggestion. get_simvar_bulk did this inline while get_aircraft_snapshot
    did not, so the same failure was actionable through one tool and opaque
    through the other; both call this now.

    `caller_units` maps the same keys read_many produces to the unit the
    caller actually asked for -- the distinction _disambiguate_not_found
    needs to tell a bad unit apart from a bad name, which read_many's
    already-resolved unit cannot supply.
    """
    errors = 0
    for key, entry in results.items():
        error_type = entry.get("error_type")
        if error_type is None:
            continue
        errors += 1
        name = key.split(":", 1)[0]
        exc_cls = _ERROR_TYPE_MAP.get(error_type, SimVarError)
        unit = caller_units.get(key) if caller_units else None
        envelope = _simvar_error_envelope(exc_cls(entry["error"]), name, unit)
        entry["error"] = envelope.message
        entry["error_code"] = envelope.error
        entry["suggestion"] = envelope.suggestion
        if envelope.suggestions:
            entry["suggestions"] = envelope.suggestions
    return len(results) - errors, errors


@handle_simconnect_errors
@require_connection(needs_accessor=True)
async def get_simvar_bulk(
    variables: Annotated[
        list[dict],
        Field(description="Variables to read. Each dict takes 'name' and optional "
                          "'unit' and 'index'. Example: "
                          '[{"name": "PLANE_LATITUDE"}, '
                          '{"name": "ENG_N1_RPM", "index": 1, "unit": "percent"}]. '
                          f"At most {MAX_BULK_VARIABLES} entries per call."),
    ],
) -> SimVarBulkResult | ToolError:
    """Read several SimVars in one call.

    Results are keyed by 'NAME' or 'NAME:index'. A failure on one variable
    does not abort the others -- that entry carries an 'error' instead.
    """
    # Deliberately no min_length/max_length on the Field above: FastMCP
    # validates a tool's arguments against its Pydantic-derived schema
    # BEFORE the function body runs, so a schema-level max_length here would
    # reject an over-limit call with a generic framework validation error
    # and this handler -- and its friendly, actionable TOO_MANY_VARIABLES
    # ToolError below -- would never run for a real MCP caller (confirmed
    # against the installed mcp SDK's FuncMetadata.call_fn_with_arg_validation,
    # which calls arg_model.model_validate(...) ahead of invoking the tool).
    # Only a direct Python call (as tests do) would ever reach it, silently
    # testing a path real callers can't take. The runtime check below is the
    # only enforcement, so it stays reachable and keeps its own error shape.
    if len(variables) > MAX_BULK_VARIABLES:
        return ToolError(
            error="TOO_MANY_VARIABLES",
            message=(
                f"Requested {len(variables)} variables; msfs_get_simvars_bulk accepts at "
                f"most {MAX_BULK_VARIABLES} per call."
            ),
            suggestion=f"Split the request into batches of {MAX_BULK_VARIABLES} or fewer.",
        )

    manager = SimConnectManager()
    # read_many only sees the resolved unit, not whether the caller supplied
    # one -- and that distinction is exactly what _disambiguate_not_found
    # (via _simvar_error_envelope) needs to tell a bad unit apart from a bad
    # name. Keep the caller-supplied unit per key so a failed entry can be
    # diagnosed after the fact.
    caller_units: dict[str, str | None] = {}
    requests = []
    for var in variables:
        name = var["name"]
        unit = var.get("unit")
        index = var.get("index")  # index 0 must survive
        key = name if index is None else f"{name}:{index}"
        caller_units[key] = unit
        requests.append((name, unit, index))

    # No timeout argument: read_many sizes the batch budget from the number
    # of requests itself (see SimVarAccessor.read_many). It used to take a
    # TOTAL budget defaulting to one variable's worth, so this call -- for
    # up to MAX_BULK_VARIABLES variables -- ran on a single read's budget.
    results = await manager.run_sync(lambda: manager.accessor.read_many(requests))

    ok_count, error_count = diagnose_bulk_entries(results, caller_units)

    return SimVarBulkResult(
        count=len(results),
        ok_count=ok_count,
        error_count=error_count,
        variables=results,
    )


@handle_simconnect_errors
@require_connection(needs_accessor=True)
async def get_object_simvars(
    object_id: Annotated[
        int,
        Field(description="SimConnect object ID to read from, as returned by "
                          "msfs_create_ai_object. Not a title and not an index.",
              ge=0),
    ],
    variables: Annotated[
        list[dict],
        Field(description="Variables to read. Each dict takes 'name' and optional "
                          "'unit' and 'index'. Example: "
                          '[{"name": "PLANE_ALTITUDE", "unit": "meters"}, '
                          '{"name": "GROUND_ALTITUDE", "unit": "meters"}]. '
                          f"At most {MAX_BULK_VARIABLES} entries per call."),
    ],
) -> SimVarObjectResult | ToolError:
    """Read SimVars from a specific SimConnect object rather than the user aircraft.

    Takes an `object_id` from msfs_create_ai_object and reads the same way
    msfs_get_simvars_bulk does, against that object. Results are keyed by
    'NAME' or 'NAME:index'; a failure on one variable does not abort the
    others.

    Two cautions that matter more than the usual ones.

    **A value here is not proof the sim maintains that variable for this
    object.** SimConnect answers a well-formed request for any object, and a
    variable the sim does not compute for AI objects can come back as a
    perfectly plausible 0.0 rather than an error. Before trusting a variable
    on spawned objects, check it somewhere the answer is already known
    independently. Do not infer from "it returned a number" that it returned
    a measurement.

    **A stale object_id is not an error either.** SimConnect will not tell
    you the object was removed, or that a flight restart cleared every AI
    object; the read simply times out or returns nothing useful. Treat an
    id as valid only for the session and the flight that produced it.

    Reading is deliberately one-way: there is no write counterpart, because
    nothing needs to set variables on arbitrary objects and a stray write
    against a wrong id is not something a caller could easily notice.
    """
    # Same reasoning as get_simvar_bulk: the cap is enforced here in the body
    # rather than as a schema max_length, so real MCP callers get this
    # ToolError instead of a generic framework validation failure.
    if len(variables) > MAX_BULK_VARIABLES:
        return ToolError(
            error="TOO_MANY_VARIABLES",
            message=(
                f"Requested {len(variables)} variables; msfs_get_object_simvars accepts "
                f"at most {MAX_BULK_VARIABLES} per call."
            ),
            suggestion=f"Split the request into batches of {MAX_BULK_VARIABLES} or fewer.",
        )

    manager = SimConnectManager()
    caller_units: dict[str, str | None] = {}
    requests = []
    for var in variables:
        name = var["name"]
        unit = var.get("unit")
        index = var.get("index")  # index 0 must survive
        key = name if index is None else f"{name}:{index}"
        caller_units[key] = unit
        requests.append((name, unit, index))

    results = await manager.run_sync(
        lambda: manager.accessor.read_many(requests, object_id=object_id)
    )

    ok_count, error_count = diagnose_bulk_entries(results, caller_units)

    return SimVarObjectResult(
        count=len(results),
        ok_count=ok_count,
        error_count=error_count,
        variables=results,
        object_id=object_id,
    )


SIMVAR_COLUMNS = [
    ("name", "Name"),
    ("units", "Units"),
    ("settable", "Settable"),
    ("category", "Category"),
    ("description", "Description"),
]


@handle_simconnect_errors
async def search_simvars(
    keyword: Annotated[str, Field(description="Search term, e.g. 'altitude', 'engine'",
                                  min_length=1, max_length=100)],
    category: Annotated[
        str | None, Field(description="Restrict to one category, e.g. 'Aircraft Position'")
    ] = None,
    limit: Annotated[int, Field(description="Maximum results", ge=1, le=200)] = DEFAULT_LIMIT,
    offset: Annotated[int, Field(description="Results to skip, for paging", ge=0)] = 0,
    response_format: Annotated[
        ResponseFormat, Field(description="'markdown' for a compact table, 'json' for rows")
    ] = ResponseFormat.MARKDOWN,
) -> SearchResult | ToolError:
    """Search the SimVar catalog by keyword.

    Returns each variable's units and whether it is settable, so you can call
    msfs_get_simvar or msfs_set_simvar with the right arguments. Results are
    paginated.
    """
    rows = search_catalog(keyword, category)
    return build_search_result(
        rows, offset, limit, response_format, SIMVAR_COLUMNS,
        title=f"SimVars matching '{keyword}'",
        query=keyword, filters={"category": category},
    )


@handle_simconnect_errors
async def list_simvar_categories() -> CategoryList | ToolError:
    """List every SimVar category with its variable count.

    Use this to discover category names for the 'category' filter on
    msfs_search_simvars.
    """
    catalog = load_catalog()
    categories = {name: len(entries) for name, entries in catalog.items()}
    return CategoryList(categories=categories, total_variables=sum(categories.values()))


@handle_simconnect_errors
@require_connection(needs_accessor=True)
async def watch_simvar(
    name: Annotated[
        str, Field(description="SimVar name to watch", min_length=1, max_length=128)
    ],
    unit: Annotated[
        str | None,
        Field(description="Unit to read in. Defaults to the catalog unit."),
    ] = None,
    index: Annotated[
        int | None,
        Field(description="Index for indexed SimVars. Index 0 is valid.", ge=0, le=64),
    ] = None,
    interval_ms: Annotated[
        int, Field(description="Polling interval in milliseconds", ge=50, le=10000)
    ] = 500,
    duration_s: Annotated[
        int, Field(description="Total sampling duration in seconds", ge=1, le=30)
    ] = 5,
) -> WatchResult | ToolError:
    """Sample a SimVar over time, returning a time series for debugging.

    Fails fast if the first read raises, rather than looping for the full
    duration on a name or unit that will never work.
    """
    duration_s = min(duration_s, 30)
    interval_s = max(interval_ms / 1000.0, 0.05)
    manager = SimConnectManager()
    resolved_unit = resolve_unit(name, unit)

    samples: list[dict] = []
    errors = 0
    start = time.monotonic()

    while (time.monotonic() - start) < duration_s:
        try:
            value = await manager.run_sync(
                lambda: manager.accessor.read(name, unit=unit, index=index)
            )
            samples.append({"t": round(time.monotonic() - start, 3), "value": value})
        except SimVarError as e:
            errors += 1
            if not samples:
                # Fail fast on a name/unit that will never work, diagnosed
                # the same way get_simvar diagnoses it. This used to always
                # report SIMVAR_NOT_READABLE with "check the name" -- wrong
                # advice for e.g. a bad unit on an otherwise valid, known
                # variable, where the name was never the problem.
                return _simvar_error_envelope(e, name, unit)
        await asyncio.sleep(interval_s)

    return WatchResult(
        name=name,
        unit=resolved_unit,
        index=index,
        samples=[WatchSample(**s) for s in samples],
        sample_count=len(samples),
        error_count=errors,
        duration_s=duration_s,
        interval_ms=interval_ms,
    )
