# Copyright 2019-2026 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.
# Author: Sandro Wenzel <sandro.wenzel@cern.ch>
# Since: 2026-08

"""Make `import OCC` work regardless of which interpreter started us.

If OCC is not importable, `ensure_occ()` re-executes the current script, or the module started
with `python3 -m`, under the aliBuild Python that pythonOCC is built against. Call it before any
`from OCC...` import.
"""

import os
import sys
from pathlib import Path

UNRESOLVED = ("cannot locate the aliBuild area that holds pythonOCC: set ALIBUILD_ARCH_ROOT to "
              "<work dir>/<architecture>, or load the O2 environment, or set ALIBUILD_WORK_DIR")

_GUARD = "O2_CSG_OCC_REEXEC"


def arch_root():
    """The aliBuild <work dir>/<architecture> directory, or None when it cannot be found.

    ALIBUILD_ARCH_ROOT wins; otherwise the directory two levels above O2_ROOT, or the one
    architecture under ALIBUILD_WORK_DIR, whichever has pythonOCC installed. Several architectures
    under ALIBUILD_WORK_DIR with no O2_ROOT candidate is an error, not a guess.
    """
    if os.environ.get("ALIBUILD_ARCH_ROOT"):
        return Path(os.environ["ALIBUILD_ARCH_ROOT"])
    candidates = []
    o2_root = Path(os.environ.get("O2_ROOT", "")).resolve()
    if os.environ.get("O2_ROOT") and len(o2_root.parents) > 1:
        candidates.append(o2_root.parents[1])
    if os.environ.get("ALIBUILD_WORK_DIR"):
        work = [p.parents[1] for p in Path(os.environ["ALIBUILD_WORK_DIR"]).glob("*/pythonOCC/latest")]
        if len(work) > 1 and not candidates:
            raise SystemExit("several aliBuild architectures hold pythonOCC ("
                             + ", ".join(sorted(p.name for p in work))
                             + "); set ALIBUILD_ARCH_ROOT to choose one")
        candidates += sorted(work)
    for candidate in candidates:
        if (candidate / "pythonOCC/latest").exists():
            return candidate
    return None


def occ_python():
    """The Python 3.10 pythonOCC is built against, or None."""
    sw = arch_root()
    return None if sw is None else sw / "Python/latest/bin/python3.10"


def occ_env_prefix():
    """The PYTHONPATH and LD_LIBRARY_PATH entries that make OCC importable, or None."""
    sw = arch_root()
    if sw is None:
        return None
    return {
        "PYTHONPATH": f"{sw}/pythonOCC/latest/lib/python3.10/site-packages:"
                      f"{sw}/Python-modules/latest/lib/python3.10/site-packages",
        "LD_LIBRARY_PATH": f"{sw}/OCCT/latest/lib:{sw}/Python/latest/lib",
    }


def have_occ() -> bool:
    try:
        import OCC  # noqa: F401
        return True
    except Exception:
        return False


def ensure_occ() -> None:
    """Re-exec this process under the pythonOCC interpreter if OCC is not importable.

    The pythonOCC paths are prepended to the inherited ones, so a process started from an O2
    shell can import both OCC and ROOT.
    """
    if have_occ():
        return
    python = occ_python()
    if python is None:
        raise SystemExit(f"OCC is not importable here, and {UNRESOLVED}")
    if os.environ.get(_GUARD):
        raise SystemExit(f"cannot import OCC even under {python}; check the pythonOCC installation")
    if not python.exists():
        raise SystemExit(f"pythonOCC interpreter not found: {python}")
    env = dict(os.environ)
    for key, prefix in occ_env_prefix().items():
        existing = env.get(key, "")
        env[key] = prefix + (":" + existing if existing else "")
    env[_GUARD] = "1"
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    if spec is not None and spec.name:
        argv = [str(python), "-m", spec.name]  # started as `python3 -m`; the working directory is kept
    else:
        argv = [str(python), str(Path(sys.argv[0]).resolve())]
    os.execve(str(python), argv + sys.argv[1:], env)
