import ast
import re
from pathlib import Path

import simconnect_mcp
from simconnect_mcp.server import mcp

# Anything shaped like a tool name in agent-facing text -- i.e.
# msfs_-prefixed -- is a claim that an agent can call that tool. This
# pattern is deliberately narrow: it must not match bare pre-rename names
# like "get_simvar", which are caught instead by the AST scan further down
# -- the whole-file scan cannot tell a genuine reference to a Python
# function (in a comment, or a private helper's docstring about internals)
# from agent-facing recovery advice.
_TOOL_NAME_PATTERN = re.compile(r"\bmsfs_[a-z][a-z0-9_]*\b")

# Measured: 90 msfs_ references across 20 files -- the 8 docs .md files,
# templates.py, and 11 tools/**/*.py modules, 7 of which hardcode a tool
# name into agent-facing text. (Was 62 across the same 20 files before the
# bare-name sweep below prefixed 27 more.)
#
# These floors exist for ONE purpose: catching a wholesale scan failure.
# Path.glob() on a renamed or moved directory returns [] with no error, so
# without them a scan that shrank to nothing would pass exactly as silently
# as a healthy one -- which is precisely how an earlier version of this test
# passed at 0 files and 0 matches with a fabricated tool name planted in the
# corpus. Genuine staleness is caught by the per-match assertion below, not
# by these numbers.
#
# So they are deliberately set BELOW the measured values. Pinning them to the
# exact current count would red the suite on any ordinary edit that merges two
# docs or rewords one suggestion= string, and a test that cries wolf on
# routine churn gets its constant nudged without thought -- including on the
# day it drops for a real reason. Measured against this headroom: renaming
# docs/ drops the scan to (12 files, 11 matches) and renaming tools/ to
# (9, 58) -- each below a floor. templates.py is appended by an explicit path
# rather than a glob, so its disappearance raises FileNotFoundError here and
# needs no floor of its own.
_MINIMUM_EXPECTED_FILES = 16
_MINIMUM_EXPECTED_MATCHES = 55


def _files_that_can_reference_a_tool() -> list[Path]:
    """Every place agent-facing text can name a tool: the embedded docs
    served as MCP resources, the prompt templates module, and the tool
    source itself -- a handful of hardcoded msfs_* names live in
    suggestion= strings there too (e.g. "reconnect with msfs_connect")."""
    pkg_dir = Path(simconnect_mcp.__file__).parent
    files = sorted((pkg_dir / "docs").glob("*.md"))
    files.append(pkg_dir / "prompts" / "templates.py")
    files.extend(sorted((pkg_dir / "tools").rglob("*.py")))
    return files


WRITE_TOOLS = {
    "msfs_set_simvar",
    "msfs_set_lvar",
    "msfs_execute_calculator_code",
    "msfs_trigger_event",
    "msfs_trigger_custom_event",
    "msfs_send_pmdg_event",
    "msfs_set_aircraft_position",
    "msfs_load_flight",
    # load_flight_plan is deliberately here, not alongside msfs_save_flight
    # below: it has no overwrite-style guard, so it stays at the
    # read_only=False default (destructive=True) -- see server.py.
    "msfs_load_flight_plan",
    "msfs_create_ai_object",
}


async def _tools():
    return {t.name: t for t in await mcp.list_tools()}


async def test_every_tool_is_msfs_prefixed():
    for name in await _tools():
        assert name.startswith("msfs_"), f"{name} lacks the service prefix"


async def test_expected_tool_count():
    assert len(await _tools()) == 33


async def test_phase_two_tools_are_registered():
    names = await _tools()
    for name in ("msfs_search_hubhop", "msfs_list_hubhop_aircraft", "msfs_load_flight",
                 "msfs_save_flight", "msfs_load_flight_plan", "msfs_create_ai_object"):
        assert name in names


async def test_consolidated_tools_replaced_their_predecessors():
    names = await _tools()
    assert "msfs_get_aircraft_snapshot" in names
    assert "msfs_browse_lvar_catalog" in names
    for gone in ("msfs_get_aircraft_state", "msfs_get_aircraft_position",
                 "msfs_get_aircraft_systems", "msfs_list_lvar_panels",
                 "msfs_list_lvar_catalogs"):
        assert gone not in names


async def test_every_tool_has_annotations_and_a_title():
    for name, tool in (await _tools()).items():
        assert tool.annotations is not None, f"{name} has no annotations"
        assert tool.annotations.title, f"{name} has no title"


async def test_every_resource_and_template_declares_mime_type_and_title():
    """Task 9 added mime_type/title to every @mcp.resource() registration
    (documentation.py's seven, state.py's two), but nothing asserted either
    field -- the exact metadata that task existed to add. Covers both
    concrete resources (list_resources) and URI templates
    (list_resource_templates), since Task 9 touched both kinds."""
    resources = await mcp.list_resources()
    templates = await mcp.list_resource_templates()
    assert resources, "expected at least one registered resource"
    assert templates, "expected at least one registered resource template"

    for r in resources:
        assert r.mimeType, f"resource {r.uri} has no mime_type"
        assert r.title, f"resource {r.uri} has no title"
    for t in templates:
        assert t.mimeType, f"resource template {t.uriTemplate} has no mime_type"
        assert t.title, f"resource template {t.uriTemplate} has no title"


async def test_write_tools_are_marked_destructive():
    tools = await _tools()
    for name in WRITE_TOOLS:
        ann = tools[name].annotations
        assert ann.readOnlyHint is False, f"{name} claims to be read-only"
        assert ann.destructiveHint is True, f"{name} is not marked destructive"


async def test_read_tools_are_marked_read_only():
    tools = await _tools()
    for name in ("msfs_get_simvar", "msfs_search_simvars", "msfs_get_aircraft_snapshot",
                 "msfs_search_events", "msfs_get_pmdg_cdu"):
        assert tools[name].annotations.readOnlyHint is True


async def test_every_tool_declares_an_output_schema():
    for name, tool in (await _tools()).items():
        assert tool.outputSchema is not None, f"{name} has no outputSchema"


async def test_no_tool_hides_its_arguments_behind_a_params_object():
    """A single Pydantic model parameter collapses the schema to {'params': ...}."""
    for name, tool in (await _tools()).items():
        assert list(tool.inputSchema.get("properties", {})) != ["params"], name


async def test_docs_and_prompts_only_reference_real_tool_names():
    """Docs, prompt templates, and hardcoded suggestion= strings in tool
    source all tell an agent (or a human reading an error message) which
    tool to call by name. A stale reference -- a rename that missed a
    file, a tool consolidated away later -- would point at something that
    no longer exists. Scanning every such file for msfs_-shaped names and
    checking each is actually registered turns the one-time rename sweep
    into a standing guarantee.

    The two floor assertions below are load-bearing, not decoration: an
    earlier version of this test asserted nothing about how many files or
    matches it found, so when the docs directory was renamed out from
    under it (reproducing that exact failure), `Path.glob()` silently
    returned `[]`, the loop below ran zero times, and the test passed at
    0-and-0 even with a fabricated unregistered tool name planted inside.
    """
    names = await _tools()
    files = _files_that_can_reference_a_tool()
    assert len(files) >= _MINIMUM_EXPECTED_FILES, (
        f"expected to scan at least {_MINIMUM_EXPECTED_FILES} files, found "
        f"only {len(files)} ({[str(f) for f in files]}) -- did a directory "
        f"get renamed or moved out from under this test?"
    )

    total_matches = 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        matches = _TOOL_NAME_PATTERN.findall(text)
        total_matches += len(matches)
        for match in matches:
            assert match in names, f"{path} references unknown tool {match!r}"

    assert total_matches >= _MINIMUM_EXPECTED_MATCHES, (
        f"expected at least {_MINIMUM_EXPECTED_MATCHES} msfs_-shaped tool "
        f"references across docs/prompts/tool source, found only "
        f"{total_matches} -- the scan may be silently covering less than "
        f"it used to"
    )


# ---------------------------------------------------------------------------
# Bare (unprefixed) tool names in agent-facing strings.
#
# The msfs_-shaped scan above deliberately cannot catch these, and its
# stated exclusion -- bare names "may legitimately remain in prose
# describing internals (a Python function name, a docstring)" -- is right
# for a comment and wrong for a `suggestion=`, which is delivered verbatim
# to the agent as recovery advice. An agent that fails a read and follows
# "Use search_simvars to find the correct name" calls a tool that does not
# exist: recovery advice that guarantees a second failure.
#
# So this scans string VALUES rather than whole files, and draws the line
# by what reaches an agent rather than by syntax:
#
#   scanned      every string literal in tools/**/*.py that is not a
#                docstring (suggestion=, message=, Field(description=),
#                _ERROR_CODES entries, and helper-returned prose such as
#                lvars._no_detection_message), plus the docstrings of
#                REGISTERED TOOL functions (FastMCP serves those as the
#                tool description), plus prompts/templates.py.
#   not scanned  comments, module docstrings, and the docstrings of
#                helpers that are not registered tools -- prose about
#                Python internals, where a bare function name is the
#                correct thing to write.
# ---------------------------------------------------------------------------


def _registered_tool_functions() -> dict[str, str]:
    """Map each registered Python function name to its MCP tool name.

    Parsed from server.py's `_register(fn, "msfs_...")` calls rather than
    assumed to be "msfs_" + the function name, because three registrations
    are not: connect_to_sim -> msfs_connect, disconnect_from_sim ->
    msfs_disconnect, and get_simvar_bulk -> msfs_get_simvars_bulk. Blindly
    prefixing that last one yields "msfs_get_simvar_bulk", which is also
    not a tool -- the same class of failure this scan exists to catch.
    """
    pkg_dir = Path(simconnect_mcp.__file__).parent
    tree = ast.parse((pkg_dir / "server.py").read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_register"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and isinstance(node.args[1], ast.Constant)
        ):
            mapping[node.args[0].id] = node.args[1].value
    return mapping


def _literal_parts(node: ast.AST):
    """(lineno, text) for a string constant, or for each literal piece of
    an f-string. Interpolated expressions are skipped -- they carry values,
    not prose."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node.lineno, node.value
    elif isinstance(node, ast.JoinedStr):
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                yield node.lineno, part.value


def _agent_facing_strings(path: Path, tool_functions: set[str]):
    """(lineno, kind, text) for every string in `path` an agent can read."""
    tree = ast.parse(path.read_text(encoding="utf-8"))

    # Docstrings are skipped wholesale EXCEPT a registered tool's own,
    # which FastMCP serves to the agent as the tool description.
    skipped: set[int] = set()
    kept: list[tuple[int, str, str]] = []

    def note_docstring(node, keep: bool) -> None:
        body = getattr(node, "body", None)
        if not body:
            return
        first = body[0]
        if not (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            return
        skipped.add(id(first.value))
        if keep:
            kept.append((first.value.lineno, "tool docstring", first.value.value))

    note_docstring(tree, keep=False)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            note_docstring(node, keep=node.name in tool_functions)
        elif isinstance(node, ast.JoinedStr):
            # ast.walk reaches an f-string's literal pieces twice: once
            # through the JoinedStr, and again as bare Constants of their
            # own. Skipping the latter keeps each string counted once, so
            # the hit list is a set of real sites rather than a list with
            # every f-string duplicated.
            for part in node.values:
                if isinstance(part, ast.Constant):
                    skipped.add(id(part))

    for node in ast.walk(tree):
        if id(node) in skipped:
            continue
        for lineno, text in _literal_parts(node):
            kept.append((lineno, "string", text))
    return kept


def _bare_name_pattern(tool_functions: dict[str, str]) -> re.Pattern:
    """Match a bare tool-function name that carries no prefix.

    The lookbehind rejects an already-correct `msfs_get_simvar` and an
    attribute access like `manager.get_lvar`; the lookahead stops
    `get_simvar` matching inside `get_simvar_bulk`.
    """
    alternatives = "|".join(
        re.escape(n) for n in sorted(tool_functions, key=len, reverse=True)
    )
    return re.compile(r"(?<![\w.])(" + alternatives + r")(?![\w])")


def _scan_for_bare_tool_names() -> list[tuple[Path, int, str, str, str]]:
    """(path, lineno, kind, bare name, correct tool name) for every hit."""
    pkg_dir = Path(simconnect_mcp.__file__).parent
    tool_functions = _registered_tool_functions()
    pattern = _bare_name_pattern(tool_functions)

    files = sorted((pkg_dir / "tools").rglob("*.py"))
    files.append(pkg_dir / "prompts" / "templates.py")

    hits = []
    for path in files:
        for lineno, kind, text in _agent_facing_strings(path, set(tool_functions)):
            for match in pattern.findall(text):
                hits.append((path, lineno, kind, match, tool_functions[match]))
    return hits


async def test_the_bare_name_scan_actually_covers_something():
    """A scan that silently covered nothing would pass exactly as quietly as
    a clean one -- the failure mode the msfs_ floors above already guard
    against, since Path.rglob on a moved directory returns [] with no
    error."""
    tool_functions = _registered_tool_functions()
    assert len(tool_functions) == len(await _tools()), (
        "server.py's _register calls and the registered tool list disagree -- "
        "the scan's name mapping is parsed from those calls, so a divergence "
        "means it is scanning for the wrong names"
    )

    pkg_dir = Path(simconnect_mcp.__file__).parent
    scanned = sum(
        len(_agent_facing_strings(path, set(tool_functions)))
        for path in sorted((pkg_dir / "tools").rglob("*.py"))
    )
    assert scanned > 100, (
        f"only {scanned} agent-facing strings found across tools/ -- did the "
        "package move, or did the docstring filter start excluding everything?"
    )


def test_agent_facing_strings_never_name_a_tool_without_its_prefix():
    """`suggestion=` is recovery advice delivered verbatim to the agent.

    Naming a tool there without the msfs_ prefix points at something that
    does not exist, so the agent's recovery attempt is guaranteed to fail a
    second time. The same holds for `message=`, `Field(description=)`, the
    _ERROR_CODES table, helper-returned prose, and a registered tool's own
    docstring, which FastMCP serves as the tool description.

    Verified to fail against the pre-fix strings: 27 hits across events.py,
    lvars.py, models.py, pmdg.py and simvars.py -- including one
    (`get_simvar_bulk`) whose real tool name is not its own name with a
    prefix bolted on.
    """
    hits = _scan_for_bare_tool_names()
    assert not hits, "agent-facing text names tools that do not exist:\n" + "\n".join(
        f"  {path.name}:{lineno} ({kind}): {bare!r} should be {tool!r}"
        for path, lineno, kind, bare, tool in hits
    )
