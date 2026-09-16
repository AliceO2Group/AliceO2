#!/bin/bash
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

# Generate an ALICE 3 geometry with o2-sim and build the ACTS tracking geometry
# from it, then check every TRK chip against its ACTS surface.
#
# Run it from a scratch directory - o2-sim writes its output into $PWD:
#   mkdir -p /tmp/acts && cd /tmp/acts
#   $O2_ROOT/share/macro/... or simply:  bash <path-to>/run_test.sh
#
# Requires an environment with O2 and ACTS (e.g. alienv enter O2/latest).

set -o errexit
set -o pipefail

nEvents=${nEvents:-1}
generator=${generator:-pythia8hi}
modules=${modules:-"A3IP TRK FT3 TF3"}

# Detector layouts. This is the configuration the ACTS chain is developed
# against: a fully cylindrical vertex detector (as in the standalone actsO2
# geometries), simplified-realistic ML/OT, segmented FT3 staves and segmented
# barrel TOFs with the endcap TOFs switched off.
layout="TRKBase.layoutVD=kIRISFullCyl"
layout+=";TRKBase.layoutMLOT=kSimplifiedRealistic"
layout+=";FT3Base.layoutFT3=kSegmentedStave"
layout+=";IOTOFBase.segmentedInnerTOF=true"
layout+=";IOTOFBase.segmentedOuterTOF=true"
layout+=";IOTOFBase.enableBackwardTOF=false"
layout+=";IOTOFBase.enableForwardTOF=false"

# o2-sim ends by fetching alignment from CCDB, which needs a valid alien token
# and aborts without one - after the geometry file has already been written.
# Skip it: the ACTS geometry is built from the ideal geometry anyway. Set
# SKIP_ALIGNMENT=0 if you do want the alignment step.
if [ "${SKIP_ALIGNMENT:-1}" = "1" ]; then
  layout+=";align-geom.mDetectors=none"
fi

echo "=== o2-sim: generating the ALICE 3 geometry ==="
o2-sim-serial-run5 -n "${nEvents}" -g "${generator}" -m ${modules} \
  --configKeyValues "${layout}" 2>&1 | tee sim_alice3.log

macroDir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# The Gen3 blueprint builder needs the geometry description that goes with the
# geometry file. The one shipped with this package matches the layout above; set
# GEN3_CONFIG to use a different geometry's gen3_geometry_config.json.
if [ -n "${GEN3_CONFIG}" ]; then
  gen3Config=${GEN3_CONFIG}
elif [ -f "${O2_ROOT}/share/Detectors/Upgrades/ALICE3/ACTS/config/gen3_geometry_config.json" ]; then
  gen3Config="${O2_ROOT}/share/Detectors/Upgrades/ALICE3/ACTS/config/gen3_geometry_config.json"
elif [ -f "${macroDir}/../config/gen3_geometry_config.json" ]; then
  gen3Config="${macroDir}/../config/gen3_geometry_config.json"   # running from a source checkout
else
  gen3Config=gen3_geometry_config.json
fi
if [ ! -f "${gen3Config}" ]; then
  echo "ERROR: no Gen3 geometry config at '${gen3Config}'." >&2
  echo "       Set GEN3_CONFIG to the gen3_geometry_config.json for this geometry." >&2
  exit 1
fi
echo "Using Gen3 geometry config: ${gen3Config}"

# The macro includes ACTS and O2 headers that are not on ROOT's search path by
# default.
export ROOT_INCLUDE_PATH="${ACTS_ROOT}/include:${EIGEN3_ROOT}/include/eigen3:${O2_ROOT}/include:${ROOT_INCLUDE_PATH}"

# Compile the macro here rather than in place: ACLiC drops its .so and dictionary
# next to the macro, which would litter the source tree.
cp "${macroDir}/CheckActsTrackingGeometry.C" .

echo "=== ACTS: building the tracking geometry and checking the sensor index ==="
root.exe -b -q "CheckActsTrackingGeometry.C+(\"o2sim_geometry.root\",\"${gen3Config}\")" \
  2>&1 | tee CheckActsTrackingGeometry.log
