#!/usr/bin/env bash
#
# This example exercises the Geant4 fast simulation route on the front absorber.
#
# A fast simulation model replaces the detailed transport through a region of
# the geometry by a function from the particle that enters it to the particles
# that leave it. The model shipped here, `toyAbsorber`, is a placeholder: it
# returns the incident particle carrying on in its direction with an
# exponentially attenuated energy. What the example demonstrates is the
# machinery, not the physics.
#
# `G4.fastSimEnvelope` names the volume the model stands in for: AFaM, the
# mother of the whole absorber. The regions Geant4 needs in order to consult the
# model are derived from it by walking its subtree and collecting the media,
# because a region can only ever be "every volume of a given material".
#
# The setup is PIPE and ABSO only, which keeps the run short and puts the
# absorber in the path of everything.

set -x

EVENTS=5
MODULES="-m PIPE ABSO"
GEN="-g pythia8pp"
# Alignment is irrelevant here and switching it off keeps the example from
# needing a CCDB connection and an alien token.
COMMON="align-geom.mDetectors=none"

# --------------------------------------------------------------- 1. reference
# Detailed transport, for comparison.
mkdir -p full && cd full
o2-sim-serial -n ${EVENTS} ${GEN} -e TGeant4 ${MODULES} -o full \
    --configKeyValues "${COMMON}" > logfull 2>&1
cd ..

# -------------------------------------------------------------------- 2. fast
# G4.fastSimModels is what switches the feature on; with it empty (the default)
# nothing about the simulation changes.
mkdir -p fast && cd fast
o2-sim-serial -n ${EVENTS} ${GEN} -e TGeant4 ${MODULES} -o fast \
    --configKeyValues "${COMMON};\
G4.fastSimModels=toyAbsorber;\
G4.fastSimEnvelope=AFaM;\
G4.fastSimMinEnergy=1.0" > logfast 2>&1
cd ..

# ----------------------------------------------------------------- 3. compare
# The model prints once at setup; if this line is missing the region name did
# not resolve to a medium and nothing was applied.
grep -h "fast simulation" fast/logfast

# Tracks written per event. The absorber normally turns one incident hadron into
# a shower, and the toy model returns a single particle instead, so the fast run
# has to produce far fewer.
for d in full fast; do
  echo "=== ${d}"
  root -l -b -q "$(dirname "$0")/countTracks.macro(\"${d}/${d}\")"
done
