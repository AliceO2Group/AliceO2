#!/usr/bin/env python3

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

"""Make an exact-surface geom.C loadable through o2-sim's external-geometry mechanism.

WORKAROUND, not a fix. `loadCADGeometryHook` JITs the macro inside a namespace and hoists only
'#' lines, so the macro's forward declaration of `o2::cad::LoadSurfaceSolid` lands in a nested
`o2` and the macro fails to compile. This replaces it with an #include of O2SurfaceSolidIO.h,
which is hoisted to global scope.

Usage: patch_exact_macro.py <geom.C> [...]   (idempotent)
"""
import sys

BLOCK = """// O2SurfaceSolidIO.h is not part of the ROOT dictionary module; declare the loader
// prototype directly (the symbol resolves from libO2CADSupport).
namespace o2
{
namespace cad
{
bool LoadSurfaceSolid(const std::string& file, O2BVHSurfaceSolid& solid);
} // namespace cad
} // namespace o2
"""

REPLACEMENT = """// PATCHED by validation/demo/patch_exact_macro.py: the emitted forward declaration is
// nested by the JIT namespace wrapper in CADGeometryUtils.cxx and shadows ::o2. A '#include'
// is hoisted to global scope by that wrapper, so it declares the right symbol.
#include "CADSupport/O2SurfaceSolidIO.h"
"""


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rc = 0
    for path in sys.argv[1:]:
        text = open(path).read()
        if REPLACEMENT.splitlines()[-1] in text:
            print(f"{path}: already patched")
            continue
        if BLOCK not in text:
            if "O2BVHSurfaceSolid" not in text:
                print(f"{path}: no exact-surface prelude, nothing to do")
            else:
                print(f"{path}: ERROR prelude not recognised -- converter output changed")
                rc = 1
            continue
        open(path, "w").write(text.replace(BLOCK, REPLACEMENT))
        print(f"{path}: patched")
    return rc


if __name__ == "__main__":
    sys.exit(main())
