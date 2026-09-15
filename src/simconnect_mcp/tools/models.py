"""Pydantic models for tool inputs and outputs.

Tools return a `SomeModel | ToolError` union so FastMCP can emit an
outputSchema covering both paths.  A bare `-> SomeModel` annotation would
fail validation whenever handle_simconnect_errors returns an error.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from simconnect_mcp.simvar_access import (
    SimVarBatchTimeoutError,
    SimVarError,
    SimVarNotFoundError,
    SimVarNotSettableError,
    SimVarTimeoutError,
    UnitMismatchError,
)


class ToolError(BaseModel):
    """Failure envelope. Field names match the pre-Phase-1 error dicts.

    Two conventions this surface holds to, both worth knowing before adding
    a code:

    * "X is not there to use" codes are spelled ``<SUBJECT>_NOT_AVAILABLE``
      (MOBIFLIGHT_NOT_AVAILABLE, FACILITIES_NOT_AVAILABLE,
      POSITION_NOT_AVAILABLE, HUBHOP_NOT_AVAILABLE). ACCESSOR_UNAVAILABLE is
      the one exception and stays as it is deliberately: it predates the
      convention and agents branch on it, and a stable machine-readable code
      is worth more than a tidy one. Do not rename it, and do not copy its
      shape for anything new.
    * ``message`` is one clause of *what went wrong*, and nothing else.
      Rationale, mitigation, and "why you are not getting a partial result"
      belong in ``suggestion``.
    """

    status: Literal["error"] = "error"
    error: str = Field(..., description="Stable machine-readable error code")
    message: str = Field(..., description="What went wrong")
    suggestion: str | None = Field(None, description="What to try next")
    suggestions: list[str] | None = Field(
        None, description="Close name matches, when the failure was a bad name"
    )


_ERROR_CODES: dict[type[SimVarError], tuple[str, str]] = {
    SimVarNotFoundError: (
        "SIMVAR_NOT_FOUND",
        "Use msfs_search_simvars to find the correct variable name.",
    ),
    SimVarNotSettableError: (
        "SIMVAR_NOT_SETTABLE",
        "This variable is read-only. Check the 'settable' flag with "
        "msfs_search_simvars, or use msfs_trigger_event for an equivalent control.",
    ),
    UnitMismatchError: (
        "UNIT_MISMATCH",
        "Check the variable's units with msfs_search_simvars, or omit the unit "
        "argument to use the catalog default.",
    ),
    SimVarTimeoutError: (
        "SIM_TIMEOUT",
        "The sim may be paused or loading. Try again shortly.",
    ),
    # Looked up by exact type, so this never shadows (nor is shadowed by)
    # its SimVarTimeoutError base above. It must not inherit that entry's
    # suggestion: retrying a batch that exhausted its own budget reproduces
    # the same result forever, and blaming a sim that was answering fine is
    # a fabricated diagnosis.
    SimVarBatchTimeoutError: (
        "BATCH_BUDGET_EXCEEDED",
        "The batch ran out of its own time budget before this variable's "
        "turn -- the sim was not necessarily stalled. Retry with fewer "
        "variables per call.",
    ),
}


def error_from(exc: Exception, suggestions: list[str] | None = None) -> ToolError:
    """Map an accessor exception to a ToolError with an actionable suggestion."""
    code, suggestion = _ERROR_CODES.get(
        type(exc), ("SIMCONNECT_ERROR", "Check that MSFS is running and try again.")
    )
    return ToolError(
        error=code, message=str(exc), suggestion=suggestion, suggestions=suggestions
    )


class Page(BaseModel):
    """Pagination metadata returned alongside every list or search result."""

    total: int = Field(..., description="Total matches before pagination")
    count: int = Field(..., description="Number of results in this response")
    offset: int = Field(..., description="Offset this response starts at")
    has_more: bool = Field(..., description="Whether more results follow")
    next_offset: int | None = Field(None, description="Offset for the next page")

    @classmethod
    def build(cls, total: int, offset: int, count: int) -> Page:
        has_more = (offset + count) < total
        return cls(
            total=total,
            count=count,
            offset=offset,
            has_more=has_more,
            next_offset=offset + count if has_more else None,
        )


class OkModel(BaseModel):
    status: Literal["ok"] = "ok"


class SimVarValue(OkModel):
    name: str
    value: float | str | None = None
    unit: str = Field(..., description="Unit the value was actually read in")
    index: int | None = None


class SimVarWriteResult(OkModel):
    name: str
    value_set: float
    unit: str
    index: int | None = None
    verified: bool = Field(
        ..., description="Whether a read-back after the write confirmed the value landed"
    )
    warning: str | None = Field(
        None,
        description="Set when verified is False -- SimConnect does not reject writes to "
        "read-only variables, so a rejected write looks identical to a successful one "
        "until read back",
    )


class SimVarBulkResult(OkModel):
    count: int
    ok_count: int = Field(
        ..., description="Variables that returned a value"
    )
    error_count: int = Field(
        ...,
        description="Variables that failed. Non-zero means part of the request did not "
        "succeed even though status is 'ok' -- inspect the individual entries",
    )
    variables: dict[str, dict[str, Any]] = Field(
        ..., description="Keyed by NAME or NAME:index; each holds a value or an error"
    )


class SimVarObjectResult(SimVarBulkResult):
    """A bulk read taken from a specific SimConnect object.

    `object_id` is echoed back so a result can never be mistaken for the
    user aircraft's. That matters more here than it looks: a variable the
    sim does not maintain for AI objects can come back as a plausible zero
    rather than an error, and the only thing distinguishing that from a
    real reading is knowing which object was asked.
    """

    object_id: int = Field(
        ..., description="The SimConnect object these variables were read from"
    )


class WatchSample(BaseModel):
    t: float = Field(..., description="Seconds since the watch started")
    value: float | str | None = None


class WatchResult(OkModel):
    name: str
    unit: str
    index: int | None = None
    samples: list[WatchSample]
    sample_count: int
    error_count: int
    duration_s: int
    interval_ms: int


class SearchResult(OkModel):
    """Search results, either rendered markdown or structured rows."""

    page: Page
    results: list[dict[str, Any]] | None = Field(
        None, description="Structured rows; null when response_format is markdown"
    )
    markdown: str | None = Field(
        None, description="Rendered table; null when response_format is json"
    )
    query: str | None = None
    filters: dict[str, Any] | None = None
    message: str | None = Field(
        None,
        description="Caveat about the results, e.g. when a catalog-scoped search could "
        "not auto-detect the aircraft and fell back to searching everything",
    )


class CategoryList(OkModel):
    categories: dict[str, int] = Field(..., description="Category name to variable count")
    total_variables: int


class EventResult(OkModel):
    event: str
    parameter: int | None = None
    resolved_via: str | None = Field(
        None, description="'catalog' or 'mapped' -- how the event name was resolved"
    )
    custom: bool = False
    message: str


class LVarValue(OkModel):
    name: str
    rpn: str
    value: float | None = None


class LVarWriteResult(OkModel):
    name: str
    value_set: float
    verified: bool | None = Field(
        ...,
        description="Tri-state. True: a read-back confirmed the value landed. False: a "
        "read-back confirmed it did not. Null: the write was sent but could not be "
        "verified -- NOT a claim that it succeeded",
    )
    warning: str | None = Field(
        None,
        description="Set when verified is False or null. SimConnect does not reject a "
        "write an aircraft simply ignores, so an unverified write looks identical to a "
        "successful one",
    )


class LVarList(OkModel):
    page: Page
    lvars: list[str]
    truncated: bool = Field(
        False,
        description="True when the raw response hit (or exceeded) the MobiFlight WASM "
        "module's ~1000-name cap. The module still reports its list as complete even "
        "when capped, so a truncated response is otherwise indistinguishable from an "
        "exhaustive one -- when true, neither 'lvars' nor 'page.total' should be read "
        "as every L-var the aircraft has registered. See 'message'. This is judged on "
        "the count alone, so an aircraft that genuinely registers exactly 1000 L-vars "
        "(or more, which the module cannot even report) would also be flagged "
        "truncated for a response that happens to be complete -- a deliberate false "
        "positive: wrongly doubting a complete list is far less harmful than wrongly "
        "trusting an incomplete one.",
    )
    message: str | None = Field(
        None,
        description="Set when 'truncated' is true: explains the ~1000-name cap and "
        "how to reach a name it may have cut off.",
    )


class CalculatorResult(OkModel):
    code: str
    mode: Literal["read", "execute"]
    value: float | None = None
    message: str | None = None


class CatalogBrowse(OkModel):
    """Catalog listing: aircraft catalogs, panels, or one panel's variables."""

    catalog: str | None = None
    page: Page
    catalogs: list[dict[str, Any]] | None = None
    panels: list[dict[str, Any]] | None = None
    panel: str | None = None
    variables: list[dict[str, Any]] | None = None
    markdown: str | None = None
    message: str | None = Field(
        None,
        description="Caveat about the results, e.g. when the aircraft catalog could not "
        "be auto-detected",
    )


class AircraftSnapshot(OkModel):
    sections: list[str] = Field(..., description="Sections included in this snapshot")
    ok_count: int = Field(0, description="Variables that returned a value")
    error_count: int = Field(
        0,
        description="Variables that failed. Non-zero means part of the snapshot did not "
        "succeed even though status is 'ok' -- inspect the individual entries",
    )
    data: dict[str, Any]


class TextResult(OkModel):
    message: str
    duration_s: float
    color: str


class PositionResult(OkModel):
    """Position/attitude after a reposition, as actually read back from the sim.

    latitude/longitude/altitude/heading/on_ground report what a post-move
    read confirmed, never the request -- a field the read-back could not
    confirm is null here (and named in `unverified`), not silently replaced
    by the requested value. `requested` carries the original request so a
    caller can compare the two, e.g. to notice the sim snapped an on-ground
    placement to terrain instead of the requested altitude.
    """

    message: str
    latitude: float | None = None
    longitude: float | None = None
    altitude: float | None = None
    heading: float | None = None
    on_ground: bool | None = None
    airspeed: int
    requested: dict[str, Any] = Field(
        ..., description="The position/attitude actually requested, for comparison"
    )
    unverified: list[str] | None = Field(
        None, description="Fields that could not be read back after the move; those "
        "fields are null above rather than echoing the request"
    )
    warning: str | None = Field(
        None, description="Set when a field was unverified, or the sim placed the "
        "aircraft somewhere other than requested (e.g. terrain-snapped altitude)"
    )


class PmdgVarResult(OkModel):
    """One PMDG SDK data field, from either the 777 or 737 NG3 catalog."""

    name: str
    value: float | int | bool | str | None = Field(
        None, description="Many PMDG SDK fields are ctypes.c_bool; bool is listed "
        "explicitly so a switch position round-trips as true/false rather than "
        "being coerced to 1.0/0.0"
    )
    display_name: str
    category: str
    catalog: str
    variant_source: str | None = Field(
        None, description="How the PMDG variant was resolved: 'explicit', "
        "'detected' (TITLE/ATC_MODEL), 'probed' (client data area response), "
        "'name_match', or 'fallback' (a guess -- not a detection)"
    )
    value_description: str | None = None
    warning: str | None = None


class PmdgCduResult(OkModel):
    """A PMDG CDU screen, as text rows and/or a structured per-cell grid."""

    cdu: int
    cdu_name: str | None = None
    powered: bool
    rows: list[str] | None = None
    grid: list[list[dict]] | None = None
    catalog: str
    variant_source: str | None = Field(
        None, description="How the PMDG variant was resolved -- see PmdgVarResult"
    )
    warning: str | None = None


class PmdgEventResult(OkModel):
    """Confirmation that a PMDG cockpit event was sent."""

    event: str
    parameter: int | None = None
    catalog: str
    variant_source: str | None = Field(
        None, description="How the PMDG variant was resolved -- see PmdgVarResult"
    )
    message: str
    warning: str | None = Field(
        None,
        description="Set when variant_source is 'fallback' or 'name_match' -- the "
        "catalog used to send this event was assumed, not detected. Unlike the read "
        "tools (a guessed catalog there just yields NO_DATA), this actually writes "
        "to the assumed SDK's control area or fires its RPN code, so a wrong guess "
        "here can reach a real, wrong aircraft system with no error at all",
    )


class FacilityInfo(OkModel):
    """One airport, waypoint, NDB or VOR looked up by ICAO identifier."""

    facility: dict[str, Any] = Field(
        ...,
        description="Parsed facility fields. Shape varies by kind -- see "
        "simconnect_mcp.facilities -- but always includes icao, region, kind, "
        "latitude, longitude, altitude_ft",
    )


class FacilityList(OkModel):
    """Airports within a radius of a point, nearest first."""

    page: Page
    center: dict[str, float] = Field(
        ..., description="Search centre used: {'latitude': ..., 'longitude': ...}"
    )
    radius_nm: float = Field(..., description="Search radius, in nautical miles")
    results: list[dict[str, Any]] | None = Field(
        None, description="Structured rows, each with a distance_nm; null when "
        "response_format is markdown"
    )
    markdown: str | None = Field(
        None, description="Rendered table; null when response_format is json"
    )


class ConnectionStatus(OkModel):
    state: str
    connected: bool
    mobiflight_available: bool
    sim_paused: bool | None = None
    sim_running: bool | None = None
    message: str | None = Field(
        None, description="Human-readable line from the manager, e.g. 'Connected to MSFS'"
    )


class FlightResult(OkModel):
    """Confirmation that a flight/flight-plan file operation completed.

    Does not arrive until MSFS is answering SimConnect again, not merely
    once the underlying save/load call itself finished -- see
    tools/flight.py's _wait_for_sim_responsive. Measured live,
    msfs_save_flight's FlightSave writes its file in a fraction of a
    second but then leaves MSFS unable to answer SimConnect at all for
    ~14s; `duration_s` reports the true cost of that instead of the
    file-exists moment, and `warning` fires only if MSFS still had not
    resumed answering once the (much longer) wait bound elapsed.
    """

    action: str = Field(
        ..., description="Which operation ran: msfs_load_flight, msfs_save_flight, or "
        "msfs_load_flight_plan"
    )
    path: str = Field(..., description="Absolute path of the file involved")
    message: str
    duration_s: float = Field(
        ..., description="Total time this call took, in seconds -- including the wait "
        "for MSFS to resume answering SimConnect after the underlying save/load. A "
        "save or load can legitimately take upwards of ten seconds; this lets a "
        "caller tell a slow-but-successful call apart from a hung one."
    )
    warning: str | None = Field(
        None,
        description="Set when MSFS had not resumed answering SimConnect requests "
        "within the wait bound after this operation completed. status is still "
        "'ok' -- the save/load itself succeeded -- but the sim may still be busy, "
        "so the very next tool call could be slow or fail.",
    )


class AiObjectResult(OkModel):
    """Confirmation that MSFS accepted an AI object spawn request.

    SimConnect_AICreateSimulatedObject's HRESULT is checked, so a rejected
    packet returns a ToolError rather than this model. But MSFS accepts a
    title that matches no installed aircraft without any error -- the
    object is simply never created -- so acceptance alone never confirmed
    anything actually appeared in the sim.

    `object_id` closes that gap when it can: it is SimConnect's own
    SIMCONNECT_RECV_ID_ASSIGNED_OBJECT_ID reply, correlated back to this
    call's request id, and only ever arrives for an object MSFS actually
    created -- unlike the initial HRESULT, which just means the request was
    queued. See `message` for which case applies.
    """

    title: str
    latitude: float
    longitude: float
    object_id: int | None = Field(
        None,
        description="SimConnect's assigned object id, present only when MSFS "
        "confirmed the object was actually created. Null covers three distinct "
        "cases collapsed into one honest 'no confirmation' answer: the reply "
        "had not arrived when this call gave up waiting, the title matched no "
        "installed aircraft (MSFS then never sends a reply at all), or this "
        "connection has no request registry to correlate one with (the plain "
        "SimConnect fallback). See `message` for which of the three applies.",
    )
    message: str
