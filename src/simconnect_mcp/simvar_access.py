"""Generic SimVar access via SimConnect data definitions.

Replaces SimConnect.AircraftRequests, which holds a hardcoded table of 828
variables each bound to one fixed unit, and signals failure by returning
None/False.  That design makes four things impossible: honouring a requested
unit, reading variables outside the table, reading string variables, and
telling a failed write from a successful one.

This layer builds definitions directly -- the native AddToDataDefinition +
SetDataOnSimObject pattern -- so all four work.

SimConnectManager.set_lvar now routes through here too (see write's
`raw_name`). It used to hold its own hand-rolled copy of the same pattern,
which meant a fresh definition ID per write, a bare UnicodeEncodeError for
a non-ASCII name, and no send-ID correlation for a rejected write.
"""
from __future__ import annotations

import ctypes
import logging
import math
import time
from collections import OrderedDict
from ctypes.wintypes import DWORD

from SimConnect.Constants import SIMCONNECT_OBJECT_ID_USER, SIMCONNECT_UNUSED
from SimConnect.Enum import (
    SIMCONNECT_DATA_SET_FLAG,
    SIMCONNECT_DATATYPE,
    SIMCONNECT_PERIOD,
)

from simconnect_mcp.data.simvar_catalog import is_string_var, resolve_unit
from simconnect_mcp.dispatch import PendingRequest

log = logging.getLogger(__name__)

STRING_SIZE = 256
DEFINITION_CACHE_SIZE = 256
DEFAULT_TIMEOUT = 2.0

# Ceiling on read_many's total _sim_lock hold time, independent of how many
# variables were requested. Without it, a batch's budget is
# `len(requests) * per_item_timeout` with nothing capping the product, which
# reaches 200s at MAX_BULK_VARIABLES=100 (tools/simvars.py) and
# DEFAULT_TIMEOUT=2.0 -- and, worse, one unresponsive variable anywhere in
# the batch could consume that entire 200s by itself (see the per-item cap
# in read_many below), starving every variable after it. Measured live, a
# healthy 100-variable read completes in 2.94s. 20s gives roughly 7x
# headroom over that measurement while capping the worst case at a tenth of
# what it was.
MAX_BATCH_BUDGET = 20.0


class SimVarError(Exception):
    """Base class for SimVar access failures."""


class SimVarNotFoundError(SimVarError):
    """SimConnect did not recognise the variable name."""


class SimVarNotSettableError(SimVarError):
    """The variable exists but cannot be written."""


class UnitMismatchError(SimVarError):
    """The requested unit is not valid for this variable."""


class SimVarTimeoutError(SimVarError):
    """No data and no exception arrived before the timeout."""


class SimVarBatchTimeoutError(SimVarTimeoutError):
    """A batch read ran out of its shared time budget on this variable.

    Deliberately distinct from SimVarTimeoutError, which means the sim
    itself failed to answer within a full per-item budget. Conflating the
    two produced a fabricated diagnosis: a batch that exhausted its own
    budget reported "the sim may be paused or loading, try again shortly",
    sending a caller into a retry loop that reproduces the identical result
    forever. Budget exhaustion is fixed by a smaller batch, not by waiting.
    """


# SimConnect reports the same exception codes for different underlying
# causes depending on the operation.  A bad unit on a read and a write to a
# read-only variable both surface as DATA_ERROR, so the mapping is chosen by
# operation rather than from one global table.  (There is no
# SIMCONNECT_EXCEPTION_ILLEGAL_OPERATION -- verify any code you add against
# SimConnect.Enum.SIMCONNECT_EXCEPTION before using it.)
_NOT_FOUND = frozenset({
    "SIMCONNECT_EXCEPTION_NAME_UNRECOGNIZED",
    # RequestDataOnSimObject reports this when its definition was never
    # successfully built -- in practice, because the variable name was bad
    # and AddToDataDefinition already raised NAME_UNRECOGNIZED for it. Once
    # both packets are bound, either exception can arrive first, so both
    # must map to the same typed error for the result to be deterministic.
    "SIMCONNECT_EXCEPTION_UNRECOGNIZED_ID",
})

# Ambiguous codes: on a read these mean the unit or datatype was wrong; on a
# write they usually mean the variable is not settable.
_DATA_CODES = frozenset({
    "SIMCONNECT_EXCEPTION_DATA_ERROR",
    "SIMCONNECT_EXCEPTION_INVALID_DATA_TYPE",
    "SIMCONNECT_EXCEPTION_INVALID_DATA_SIZE",
    "SIMCONNECT_EXCEPTION_DEFINITION_ERROR",
    "SIMCONNECT_EXCEPTION_OUT_OF_BOUNDS",
})


def simconnect_name(name: str, index: int | None = None) -> bytes:
    """Convert an MCP-style SimVar name to the form SimConnect expects.

    'PLANE_ALTITUDE'      -> b'PLANE ALTITUDE'
    'ENG_N1_RPM', index=2 -> b'ENG N1 RPM:2'

    `index` is compared against None, not truthiness: index 0 is valid.
    """
    base = name.strip().upper().split(":", 1)[0].replace("_", " ")
    if index is not None:
        base = f"{base}:{index}"
    return base.encode("ascii")


def values_match(readback: float, written: float) -> bool:
    """Whether a post-write read-back confirms the value landed.

    One definition of "landed" for every write path, so a SimVar write and
    an L-var write cannot drift into two different tolerances.
    """
    return math.isclose(readback, written, rel_tol=1e-6, abs_tol=1e-6)


class SimVarAccessor:
    """Reads and writes SimVars through SimConnect data definitions."""

    def __init__(self, sm) -> None:
        self._sm = sm
        self._definitions: OrderedDict[tuple[str, str, int | None, bool], int] = OrderedDict()

    @staticmethod
    def _cache_key(
        name: str, unit: str, index: int | None, is_string: bool, raw: bool = False
    ) -> tuple[str, str, int | None, bool, bool]:
        """Cache identity for one data definition.

        A SimVar name is case-insensitive and is folded to upper case so
        'plane_altitude' and 'PLANE_ALTITUDE' share one definition. A RAW
        datum name is not folded: measured against a live MSFS 2024 build,
        L-var lookup happened to be case-insensitive there, but the MSFS
        documentation calls L-var names case-sensitive, and the two
        mistakes are not symmetric -- keeping the case costs at most one
        extra definition slot for a caller that spells the same variable
        two ways, while folding it would hand two genuinely different
        variables the same definition. `raw` is part of the key so a raw
        datum and a SimVar can never collide on the name alone.
        """
        return (
            name.strip() if raw else name.strip().upper(),
            unit,
            index,
            is_string,
            raw,
        )

    def definition_id(
        self,
        name: str,
        unit: str,
        index: int | None,
        is_string: bool,
        raw_name: bool = False,
    ) -> tuple[int, int | None]:
        """Definition ID for this (name, unit, index), creating it once.

        Returns (definition_id, send_id). send_id is the packet ID of the
        AddToDataDefinition call when one was actually issued (cache miss);
        it is None on a cache hit. AddToDataDefinition can itself raise
        NAME_UNRECOGNIZED for a bad variable name -- or a DATA_ERROR-style
        code for a bad unit -- and that exception correlates to its own
        packet, not to the RequestDataOnSimObject that follows, so callers
        must bind this send_id too or the exception is never matched.

        Definition IDs are a finite SimConnect resource and new_def_id()
        rebuilds an Enum on every call, so they must not be created per read.
        Eviction here drops only our own cache mapping -- SimConnect
        definitions ARE reclaimable (dll.ClearDataDefinition is bound in
        SimConnect.Attributes, and the vendored RequestList.redefine() calls
        it), but this cache does not call it, so an evicted definition stays
        registered with SimConnect for the life of the connection. The cache
        bound therefore caps how many distinct (name, unit, index)
        combinations we track at once; it does not free the underlying
        SimConnect-side resource.

        `unit` doubles as part of the cache key even for string variables,
        where the caller passes the literal sentinel "string" rather than a
        real SimConnect unit -- that keeps a string definition and a numeric
        one for the same variable name distinct in `_definitions`. But a
        STRING256 definition must be registered with a NULL unit: passing
        b"string" to AddToDataDefinition makes SimConnect raise
        NAME_UNRECOGNIZED (verified against a live sim, where it made every
        string SimVar, including TITLE, unreadable). So the sentinel is
        substituted with None right at the DLL call below, and never leaks
        into the cache key.

        `name`/`unit` are encoded to ASCII up front, before new_def_id() is
        even called, so a non-ASCII value (e.g. get_simvar(..., unit="°"))
        raises the same typed SimVarNotFoundError/UnitMismatchError the tool
        layer already knows how to report -- rather than a raw
        UnicodeEncodeError that handle_simconnect_errors' generic `except
        Exception` turns into an opaque UNEXPECTED envelope -- and never
        burns a definition slot on a value that was never going to work.

        `raw_name=True` passes `name` to AddToDataDefinition verbatim
        instead of putting it through simconnect_name(). An L-var write
        needs that: the datum name is 'L:SOME_VAR', and simconnect_name --
        which upper-cases, splits on the first ':' and turns underscores
        into spaces, all correct for a SimVar -- would reduce it to 'L'.
        Everything else about the definition is identical, which is the
        whole reason L-var writes route through here rather than through a
        parallel hand-rolled path: one definition cache, one ASCII check,
        one send-ID correlation.
        """
        key = self._cache_key(name, unit, index, is_string, raw_name)
        cached = self._definitions.get(key)
        if cached is not None:
            self._definitions.move_to_end(key)
            return cached, None

        try:
            encoded_name = (
                name.strip().encode("ascii") if raw_name
                else simconnect_name(name, index)
            )
        except UnicodeEncodeError as e:
            raise SimVarNotFoundError(
                f"Variable name '{name}' is not valid ASCII: {e}"
            ) from e

        encoded_unit = None
        if not is_string:
            try:
                encoded_unit = unit.encode("ascii")
            except UnicodeEncodeError as e:
                raise UnitMismatchError(
                    f"Unit '{unit}' is not valid ASCII: {e}"
                ) from e

        def_id = self._sm.new_def_id().value
        datatype = (
            SIMCONNECT_DATATYPE.SIMCONNECT_DATATYPE_STRING256
            if is_string
            else SIMCONNECT_DATATYPE.SIMCONNECT_DATATYPE_FLOAT64
        )
        self._sm.dll.AddToDataDefinition(
            self._sm.hSimConnect,
            def_id,
            encoded_name,
            encoded_unit,
            datatype,
            ctypes.c_float(0.0),
            SIMCONNECT_UNUSED,
        )
        send_id = self._last_packet_id()
        self._definitions[key] = def_id
        if len(self._definitions) > DEFINITION_CACHE_SIZE:
            self._definitions.popitem(last=False)
        return def_id, send_id

    def _evict(self, key: tuple[str, str, int | None, bool]) -> None:
        """Drop a cached definition after SimConnect rejects it.

        Without this, a definition that failed (bad name, bad unit) stays
        cached and every retry silently reuses the same broken definition
        instead of re-registering -- so a fixed unit or a renamed variable
        could never succeed on retry.
        """
        self._definitions.pop(key, None)

    def _last_packet_id(self) -> int:
        """Packet ID of the call just sent, for exception correlation."""
        out = DWORD(0)
        self._sm.dll.GetLastSentPacketID(self._sm.hSimConnect, out)
        return out.value

    def _raise_for(
        self, exception_name: str, name: str, unit: str, writing: bool = False
    ) -> None:
        """Translate a SimConnect exception into a typed error."""
        if exception_name in _NOT_FOUND:
            raise SimVarNotFoundError(f"SimConnect does not recognise SimVar '{name}'")
        if exception_name in _DATA_CODES:
            if writing:
                raise SimVarNotSettableError(
                    f"SimConnect rejected the write to '{name}'. It is most likely "
                    f"read-only; the unit '{unit}' or the value may also be invalid "
                    f"({exception_name})."
                )
            raise UnitMismatchError(
                f"Unit '{unit}' is not valid for SimVar '{name}' ({exception_name})"
            )
        raise SimVarError(f"SimConnect rejected '{name}': {exception_name}")

    def read(
        self,
        name: str,
        unit: str | None = None,
        index: int | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        raw_name: bool = False,
        object_id: int = SIMCONNECT_OBJECT_ID_USER,
    ):
        """Read one SimVar. Returns float, or str for string variables.

        Raises SimVarNotFoundError, UnitMismatchError, SimVarTimeoutError.

        `object_id` selects WHICH object the variable is read from, and
        defaults to the user aircraft so every existing caller is unchanged.
        SimVars are not properties of the user's aircraft -- they are read on
        an object, and RequestDataOnSimObject has always taken an id -- this
        was simply pinned to the user by a literal. An id from
        create_ai_object reads that spawned object instead.

        The definition cache is deliberately NOT keyed by object_id: a data
        definition describes a variable layout, not an object, and the same
        definition is valid for every object. Keying it per object would
        allocate a fresh definition per spawned probe and leak them.

        `raw_name=True` reads a verbatim datum name such as 'L:SOME_VAR'.
        The SimVar catalog knows nothing about those, so neither
        is_string_var nor resolve_unit is consulted: the value is numeric
        and the unit defaults to 'number', which is what AddToDataDefinition
        wants for an L-var.
        """
        as_string = False if raw_name else is_string_var(name)
        if raw_name:
            resolved_unit = unit or "number"
        else:
            resolved_unit = "string" if as_string else resolve_unit(name, unit)
        key = self._cache_key(name, resolved_unit, index, as_string, raw_name)

        req_id = self._sm.registry.acquire_request_id(lambda: self._sm.new_request_id().value)
        pending = PendingRequest(request_id=req_id, is_string=as_string)

        # register() runs inside this try (rather than before it) so that
        # ANYTHING raised before the request completes -- including
        # definition_id()'s ASCII encoding of a caller-supplied name/unit --
        # still reaches the finally below and discards the registry entry.
        # A registration that outlives its request leaks permanently: the
        # request ID is never freed for reuse and, if a send ID had already
        # been bound, that entry is never removed from `_by_send` either.
        try:
            self._sm.registry.register(pending)

            # The lock spans BOTH sends: on a cache miss, AddToDataDefinition
            # can itself raise NAME_UNRECOGNIZED (bad variable) or a
            # DATA_ERROR-style code (bad unit), and that exception correlates
            # to its own packet, not to the RequestDataOnSimObject that
            # follows. Binding only the request's send ID would leave a
            # definition-level exception unmatched, and the caller would see
            # a timeout instead of the correct typed error.
            with self._sm.registry.pending_lock:
                def_id, def_send_id = self.definition_id(
                    name, resolved_unit, index, as_string, raw_name
                )
                if def_send_id is not None:
                    self._sm.registry.bind_send_id(pending, def_send_id, _locked=True)

                self._sm.dll.RequestDataOnSimObject(
                    self._sm.hSimConnect,
                    req_id,
                    def_id,
                    object_id,
                    SIMCONNECT_PERIOD.SIMCONNECT_PERIOD_ONCE,
                    0,
                    0,
                    0,
                    0,
                )
                self._sm.registry.bind_send_id(pending, self._last_packet_id(), _locked=True)

            if not pending.done.wait(timeout):
                raise SimVarTimeoutError(
                    f"No response for SimVar '{name}' within {timeout}s. "
                    "The sim may be paused, loading, or not running."
                )
            if pending.exception is not None:
                # A definition that SimConnect just rejected must not
                # survive in the cache, or every retry reuses the same
                # broken definition and can never succeed.
                self._evict(key)
                self._raise_for(pending.exception, name, resolved_unit)
            return pending.value
        finally:
            self._sm.registry.discard(pending)

    def read_many(
        self,
        requests: list[tuple[str, str | None, int | None]],
        per_item_timeout: float = DEFAULT_TIMEOUT,
        object_id: int = SIMCONNECT_OBJECT_ID_USER,
    ) -> dict[str, dict]:
        """Read several SimVars, isolating failures.

        `object_id` is passed through to read() unchanged and defaults to the
        user aircraft, so existing callers are unaffected.

        Keys are `NAME` or `NAME:index`, so indexed variables stay distinct.
        A failure on one variable never aborts the batch.

        `per_item_timeout` is the budget for ONE variable. The batch's
        nominal budget is `len(requests) * per_item_timeout`, capped at
        MAX_BATCH_BUDGET and turned into a single deadline computed once
        here. Once the deadline has passed, remaining entries are reported
        without even attempting SimConnect.

        This parameter is deliberately per-item rather than the total it
        used to be. As a total it was impossible to get right at the call
        site: it defaulted to DEFAULT_TIMEOUT (sized for a single read), and
        two of the three callers passed nothing -- so a 100-variable
        get_simvar_bulk and a 44-variable snapshot both tried to fit inside
        one variable's budget. Measured live on an idle sim, a 100-variable
        bulk read finished 71 variables and reported the other 29 as
        failures purely because the batch budget had run out.

        Two bounds work together to keep lock-hold time under control, and
        both are needed -- neither alone is enough:

        * **The shared deadline** (`budget`/`deadline` below) means every
          read draws down one shrinking allowance rather than each waiting
          its own full timeout independently -- otherwise `_sim_lock`
          hold time would be unbounded from the caller's side.
        * **The per-item cap** (`this_read` below) means no single variable
          -- wherever it falls in the batch -- can spend more than
          `per_item_timeout` of that shared allowance. Without it, a
          variable early in the batch that hangs is simply handed whatever
          of the (possibly large) shared deadline remains as its own
          timeout, so it alone can exhaust the entire budget and starve
          every variable queued after it: at MAX_BULK_VARIABLES=100 and
          DEFAULT_TIMEOUT=2.0, that budget is 200s. Capped, that same hang
          costs at most `per_item_timeout` and the rest of the batch still
          gets its turn.

        An entry that misses out on time reports SimVarBatchTimeoutError,
        never the plain SimVarTimeoutError -- a batch running out of its own
        budget is not the sim stalling, and must not be diagnosed as one.
        Only a read that had a full `per_item_timeout` to itself and still
        got no answer is reported as a genuine sim timeout.
        """
        results: dict[str, dict] = {}
        total = len(requests)
        budget = min(total * per_item_timeout, MAX_BATCH_BUDGET)
        deadline = time.monotonic() + budget
        for position, (name, unit, index) in enumerate(requests):
            key = name if index is None else f"{name}:{index}"
            resolved = resolve_unit(name, unit)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                results[key] = {
                    "error": (
                        f"The batch read budget of {budget:.1f}s ran out after "
                        f"{position} of {total} variable(s), before reaching "
                        f"'{name}'. The batch ran out of time; this says nothing "
                        f"about whether the sim is responding."
                    ),
                    "error_type": SimVarBatchTimeoutError.__name__,
                    "unit": resolved,
                }
                continue
            # Capped at per_item_timeout so no single variable -- wherever
            # it falls in the batch -- can spend more than its own share of
            # the shared deadline, even when far more than that remains.
            this_read = min(remaining, per_item_timeout)
            try:
                results[key] = {"value": self.read(name, unit, index, this_read,
                                                   object_id=object_id),
                                "unit": resolved}
            except SimVarTimeoutError as e:
                # A read that got less than a full per-item share was
                # squeezed by the batch, not stalled by the sim. Only a read
                # that had its whole share and still got nothing is a real
                # sim timeout.
                #
                # Two independent ways a read can be squeezed below its own
                # per_item_timeout, either of which makes `remaining <
                # per_item_timeout` a real signal rather than noise:
                #
                # * `position > 0`: a predecessor spent part of the shared
                #   budget before this item's turn.
                # * `budget < per_item_timeout`: the MAX_BATCH_BUDGET cap
                #   (`:438` above) left the WHOLE batch less than one item's
                #   nominal share, so even the first item never had a full
                #   per_item_timeout available. total * per_item_timeout is
                #   only the budget when it is under the cap -- a single-item
                #   batch with per_item_timeout=30 gets
                #   budget=min(30, MAX_BATCH_BUDGET)=20, for example. No
                #   production call site passes a per-item timeout above
                #   MAX_BATCH_BUDGET today, so this path is reachable but
                #   untested outside test_simvar_access.py.
                #
                # Neither applies to position 0 in an uncapped batch (budget
                # >= per_item_timeout there): its `remaining` reads a few
                # microseconds below per_item_timeout purely from the cost of
                # computing the deadline, and that timing artifact must NOT
                # be misreported as budget exhaustion -- only a genuine
                # squeeze should be.
                if (
                    (position > 0 or budget < per_item_timeout)
                    and remaining < per_item_timeout
                ):
                    results[key] = {
                        "error": (
                            f"The batch read budget of {budget:.1f}s left only "
                            f"{remaining:.2f}s for '{name}', which was not enough "
                            f"for a reply. The batch ran out of time; this says "
                            f"nothing about whether the sim is responding."
                        ),
                        "error_type": SimVarBatchTimeoutError.__name__,
                        "unit": resolved,
                    }
                else:
                    results[key] = {"error": str(e), "error_type": type(e).__name__,
                                    "unit": resolved}
            except SimVarError as e:
                results[key] = {"error": str(e), "error_type": type(e).__name__,
                                "unit": resolved}
        return results

    def write(
        self,
        name: str,
        value: float,
        unit: str | None = None,
        index: int | None = None,
        grace: float = 0.15,
        verify: bool = False,
        raw_name: bool = False,
    ) -> bool | None:
        """Write one numeric SimVar.

        SimConnect sends no acknowledgement for a successful write -- success
        or failure alike.  It DOES still raise an exception for a bad
        variable name or a bad unit, correlated to the packet that caused it
        (see below), and that much this function catches: no exception
        within `grace` was, historically, the only signal available, and it
        remains the default ("no exception" is all `verify=False` reports,
        via a `None` return).

        Live testing turned up the part the old AircraftRequests.set()-based
        code, and this function's first version, both got wrong: SimConnect
        does not raise anything when the name and unit are both fine but the
        variable simply will not accept a write.  AIRSPEED_TRUE -- a
        calculated, read-only value -- silently ignores a write with no
        exception of any kind; read-back confirms the value never changed.
        The catalogue's `settable` flag cannot substitute for this check --
        it is wrong in both directions on live data (AIRSPEED_TRUE is marked
        settable=True; AUTOPILOT_ALTITUDE_LOCK_VAR, which genuinely accepts
        writes, is marked settable=False) -- so it must not gate writes here.

        Pass `verify=True` to get a real answer: after the grace window,
        this re-reads the variable in the same unit and returns True if it
        now matches `value` (within floating-point tolerance) or False if it
        does not. A False return is a report, not a judgement -- writing a
        value the sim immediately overrides (e.g. altitude in flight) or one
        the variable already held are both legitimate outcomes, so a
        mismatch is never raised as an error. If the verifying read itself
        raises (e.g. the sim disconnects between the write and the
        read-back), that exception propagates -- real information, not
        something to swallow.

        Raises SimVarNotFoundError or SimVarNotSettableError-family errors
        only for a bad name or a bad unit, exactly as before; `verify` does
        not change what can raise, only what a clean return means.

        `raw_name=True` writes a verbatim datum name such as 'L:SOME_VAR'
        (see definition_id). The SimVar catalog does not know those names,
        so resolve_unit is skipped and the unit defaults to 'number'.
        """
        if raw_name:
            resolved_unit = unit or "number"
        else:
            resolved_unit = resolve_unit(name, unit)
        key = self._cache_key(name, resolved_unit, index, False, raw_name)

        pending = PendingRequest(request_id=None)
        payload = (ctypes.c_double * 1)(float(value))

        # register() runs inside this try for the same reason as in read():
        # anything raised before the write completes -- including
        # definition_id()'s ASCII encoding of a caller-supplied name/unit,
        # or an exception from either DLL call below -- must still reach the
        # finally and discard the registry entry, or a send ID bound by the
        # first call (AddToDataDefinition) leaks forever if the second call
        # (SetDataOnSimObject) is what raises.
        try:
            self._sm.registry.register(pending)

            # The lock spans BOTH sends, exactly as in read():
            # AddToDataDefinition can itself raise NAME_UNRECOGNIZED for a
            # bad variable, correlated to its own packet rather than the
            # SetDataOnSimObject that follows. `definition_id` returns
            # (def_id, send_id_or_None) -- None on a cache hit.
            with self._sm.registry.pending_lock:
                def_id, def_send_id = self.definition_id(
                    name, resolved_unit, index, False, raw_name
                )
                if def_send_id is not None:
                    self._sm.registry.bind_send_id(pending, def_send_id, _locked=True)
                self._sm.dll.SetDataOnSimObject(
                    self._sm.hSimConnect,
                    def_id,
                    SIMCONNECT_OBJECT_ID_USER,
                    SIMCONNECT_DATA_SET_FLAG.SIMCONNECT_DATA_SET_FLAG_DEFAULT,
                    0,
                    ctypes.sizeof(ctypes.c_double),
                    ctypes.cast(payload, ctypes.c_void_p),
                )
                self._sm.registry.bind_send_id(pending, self._last_packet_id(), _locked=True)

            # An exception, if any, arrives within one dispatch round trip.
            if pending.done.wait(grace) and pending.exception is not None:
                self._evict(key)
                self._raise_for(pending.exception, name, resolved_unit, writing=True)
        finally:
            self._sm.registry.discard(pending)

        if not verify:
            return None

        # Read-back is the only reliable signal that a non-settable variable
        # silently ignored the write -- SimConnect raises nothing for that
        # case. A mismatch is reported, never raised: the sim overriding a
        # value, or the write being a no-op, are both legitimate outcomes.
        # A failure in this read (timeout, disconnect) is real information
        # and is allowed to propagate rather than being folded into False.
        readback = self.read(name, unit=resolved_unit, index=index, raw_name=raw_name)
        return values_match(readback, value)
