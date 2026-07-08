#!/usr/bin/env python
"""Document differences between the ag_comparison PRMS control files.

Self-contained: the comparisons we care about are listed in ``COMPARISONS``
below. Run it with no arguments to (re)generate the Markdown reports under
``ag_comparison/diffs/``:

    python diff_controls.py

Add a new comparison by appending a ``(control_a, control_b, output_name)``
tuple to ``COMPARISONS``.

How it works
------------
The core comparison uses pyPRMS's ``ControlFile.diff`` (following
``autotest/test_domain_subset.py`` for building a ``ControlFile``). pyPRMS's
bundled metadata does not model the GSFLOW ag / experiment control variables
(e.g. ``iter_aet_flag``, ``AET_cbh_file``, ``forcing_check_flag``), and it
raises on any unknown variable. It also models ``param_file`` as a scalar,
which cannot parse GSFLOW's multi-file form.

Rather than work around these, we extend a deep copy of the metadata
(``pws.constants.pyprms_meta``) so every variable in the files can be read
(Option A; the first-party fix would be to add these upstream to pyPRMS's
``xml/control.xml``). Each control block already declares its value type and
count, so we synthesize metadata entries directly from the files:

- unknown variable  -> add ``{datatype, context}`` inferred from the file;
- known scalar variable that appears multi-valued (e.g. ``param_file``)
  -> promote its ``context`` to ``array`` (datetime vars are left alone).

Synthesized and promoted variables are listed in each report, documenting
exactly what pyPRMS does not yet model.
"""

import copy
import pathlib as pl

import numpy as np
import pyPRMS as pp

import pywatershed as pws

HERE = pl.Path(__file__).resolve().parent
REPO = HERE.parent
DIFF_DIR = HERE / "diffs"

# (control_a, control_b, output_markdown_name). Add comparisons here as the
# adapted fgr_ag_2yr controls are created.
COMPARISONS = [
    (
        HERE / "original_control_files/nhm_ic.control",
        HERE / "original_control_files/nhm_dynamic_2000_2020.control",
        "orig__ic_vs_dynamic.md",
    ),
    (
        REPO / "test_data/fgr_ag_2yr/spinup.control",
        REPO / "test_data/fgr_ag_2yr/analysis.control",
        "fgr_ag_2yr__spinup_vs_analysis.md",
    ),
    (
        HERE / "01_gsflow/spinup_2000.control",
        HERE / "01_gsflow/analysis_2001.control",
        "01_gsflow__spinup_vs_analysis.md",
    ),
]

# PRMS control value type code -> pyPRMS datatype string.
VALUETYPE_TO_DATATYPE = {1: "int32", 2: "float32", 3: "float64", 4: "string"}


def parse_control(text: str):
    """Parse PRMS control text into (description, [block, ...]).

    Each block is a dict with keys ``name``, ``vtype`` (int), and ``values``
    (list of raw string values). The PRMS format is a leading description line,
    then ``####``-delimited blocks of ``name / numval / valuetype / value*``.
    """
    lines = text.splitlines()
    description = lines[0] if lines else ""
    raw_blocks: list[list[str]] = []
    cur: list[str] = []
    for line in lines[1:]:
        if line.strip() == "####":
            if cur:
                raw_blocks.append(cur)
            cur = []
        else:
            cur.append(line)
    if cur:
        raw_blocks.append(cur)

    blocks = []
    for b in raw_blocks:
        numval = int(b[1])
        blocks.append(
            {"name": b[0], "vtype": int(b[2]), "values": b[3 : 3 + numval]}
        )
    return description, blocks


def build_metadata(*block_lists):
    """Deep-copy pyprms_meta and extend it to cover every var in the files.

    Returns (metadata, synthesized, promoted) where synthesized/promoted map
    var name -> its (new/adjusted) metadata entry.
    """
    meta = copy.deepcopy(pws.constants.pyprms_meta)
    ctl = meta["control"]

    # Collapse each var to (valuetype, max numval seen) across all files.
    seen: dict[str, tuple[int, int]] = {}
    for blocks in block_lists:
        for blk in blocks:
            name, vt, nv = blk["name"], blk["vtype"], len(blk["values"])
            if name in seen:
                seen[name] = (seen[name][0], max(seen[name][1], nv))
            else:
                seen[name] = (vt, nv)

    synthesized: dict[str, dict] = {}
    promoted: dict[str, dict] = {}
    for name, (vt, nv) in seen.items():
        datatype = VALUETYPE_TO_DATATYPE.get(vt, "string")
        if name not in ctl:
            context = "scalar" if nv in (1, 6) else "array"
            ctl[name] = {"datatype": datatype, "context": context}
            synthesized[name] = ctl[name]
        elif (
            nv > 1
            and ctl[name].get("context") == "scalar"
            and ctl[name].get("datatype") != "datetime"
        ):
            ctl[name] = {**ctl[name], "context": "array"}
            promoted[name] = ctl[name]

    return meta, synthesized, promoted


def compare(path_a: pl.Path, path_b: pl.Path):
    """Return (diff_result, synthesized, promoted) for two control files."""
    _, blocks_a = parse_control(path_a.read_text())
    _, blocks_b = parse_control(path_b.read_text())
    meta, synthesized, promoted = build_metadata(blocks_a, blocks_b)

    cf_a = pp.ControlFile(filename=path_a, metadata=meta, verbose=False)
    cf_b = pp.ControlFile(filename=path_b, metadata=meta, verbose=False)
    result = cf_a.diff(cf_b)
    return result, synthesized, promoted


def _bullets(items) -> list[str]:
    return [f"- `{v}`" for v in items] or ["- _(none)_"]


def _is_seq(v) -> bool:
    return isinstance(v, (np.ndarray, list, tuple))


def _as_list(v) -> list[str]:
    """Coerce a (possibly array) control value to a list of strings."""
    if _is_seq(v):
        return [str(x) for x in v]
    return [str(v)]


def _set_diff_lines(var, a_val, b_val) -> list[str]:
    """Render one array-valued difference as removed/added set differences."""
    a, b = _as_list(a_val), _as_list(b_val)
    set_a, set_b = set(a), set(b)
    removed = [x for x in a if x not in set_b]  # A only, in A order
    added = [x for x in b if x not in set_a]  # B only, in B order

    lines = [f"### `{var}`  (A: {len(a)}, B: {len(b)})", ""]
    if not removed and not added:
        lines.append("- _same set; order or length differs_")
    else:
        if removed:
            lines.append(
                "- removed (A only): " + ", ".join(f"`{x}`" for x in removed)
            )
        if added:
            lines.append(
                "- added (B only): " + ", ".join(f"`{x}`" for x in added)
            )
    lines.append("")
    return lines


def format_report(name_a, name_b, result, synthesized, promoted) -> str:
    """Render the diff result as a Markdown report."""
    diffs = result["diffs"]
    scalar_diffs = {
        v: d
        for v, d in diffs.items()
        if not (_is_seq(d["self"]) or _is_seq(d["other"]))
    }
    seq_diffs = {v: d for v, d in diffs.items() if v not in scalar_diffs}

    lines = [
        f"# Control diff: `{name_a}` (A) vs. `{name_b}` (B)",
        "",
        "_Generated by `diff_controls.py` via `pyPRMS.ControlFile.diff`, with "
        "metadata extended for GSFLOW ag / multi-valued control vars (see "
        "bottom)._",
        "",
        f"## Only in A (`{name_a}`)",
        "",
        *_bullets(sorted(result["self_not_other"])),
        "",
        f"## Only in B (`{name_b}`)",
        "",
        *_bullets(sorted(result["other_not_self"])),
        "",
        "## Differing scalar values (present in both)",
        "",
    ]
    if scalar_diffs:
        lines += ["| Variable | A | B |", "| --- | --- | --- |"]
        for v in sorted(scalar_diffs):
            d = scalar_diffs[v]
            lines.append(f"| `{v}` | {d['self']} | {d['other']} |")
    else:
        lines.append("_(none)_")

    lines += ["", "## Differing list values (set differences)", ""]
    if seq_diffs:
        for v in sorted(seq_diffs):
            d = seq_diffs[v]
            lines += _set_diff_lines(v, d["self"], d["other"])
    else:
        lines.append("_(none)_")

    lines += [
        "",
        "## Metadata extensions (not modeled by pyPRMS)",
        "",
        "_Variables the script had to add to (synthesized) or adjust in "
        "(promoted to array) the pyPRMS metadata so the files could be read._",
        "",
    ]
    if synthesized or promoted:
        lines += [
            "| Variable | datatype | context | action |",
            "| --- | --- | --- | --- |",
        ]
        for v in sorted(synthesized):
            m = synthesized[v]
            lines.append(
                f"| `{v}` | {m['datatype']} | {m['context']} | synthesized |"
            )
        for v in sorted(promoted):
            m = promoted[v]
            lines.append(
                f"| `{v}` | {m['datatype']} | {m['context']} | promoted |"
            )
    else:
        lines.append("_(none)_")
    lines.append("")
    return "\n".join(lines)


def main():
    DIFF_DIR.mkdir(exist_ok=True)
    for path_a, path_b, out_name in COMPARISONS:
        if not path_a.exists() or not path_b.exists():
            missing = path_a if not path_a.exists() else path_b
            print(f"SKIP (missing): {missing}")
            continue
        result, synthesized, promoted = compare(path_a, path_b)
        report = format_report(
            path_a.name, path_b.name, result, synthesized, promoted
        )
        out_path = DIFF_DIR / out_name
        out_path.write_text(report)
        n_diff = len(result["diffs"])
        n_only = len(result["self_not_other"]) + len(result["other_not_self"])
        print(
            f"Wrote {out_path.relative_to(HERE)}  "
            f"({n_diff} diffs, {n_only} only-in-one, "
            f"{len(synthesized) + len(promoted)} metadata ext.)"
        )


if __name__ == "__main__":
    main()
