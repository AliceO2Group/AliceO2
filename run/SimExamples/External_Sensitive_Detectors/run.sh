#!/usr/bin/env bash
#
# Minimal example injecting two artificial *sensitive* external detectors into o2-sim.
#
# Neither detector is compiled into O2: both are described purely by data. The geometry of
# each is built at runtime from a ROOT macro (here hand-written stand-ins for what
# O2_CADtoTGeo.py produces from CAD files), and each is tied to a free DetID slot so its hits
# are written like those of any built-in detector -- including in parallel (multi-worker) mode,
# where the hit merger instantiates a matching receiver per detector.
#
#   ACYL  : thin silicon barrel cylinder, DetID slot ITS, built-in entrance/exit action
#   BDISK : thin silicon endcap disk,     DetID slot TST, custom JITed sensitive action
#
# Both share the single generic hit type o2::ext::Hit. Multiplicity is data-driven: add more
# entries (on more free DetID slots) to externalDetectors.json and detectorlist.json.

set -x

# run from this example's directory so the relative macro/JSON paths resolve
cd "$(dirname "$0")"

NWORKERS=2
EVENTS=5

o2-sim -j ${NWORKERS} -n ${EVENTS} -g boxgen \
       --detectorList EXTEXAMPLE:detectorlist.json \
       --extGeomFile externalDetectors.json \
       --configKeyValues 'BoxGun.number=50' \
       > sim.log 2>&1

# count and locate the produced hits (one file per detector / DetID slot)
root -l -b -q inspect_hits.macro
