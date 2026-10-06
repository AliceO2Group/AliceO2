// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_SIMSETUP_VECGEOMCHECKS_H_
#define O2_SIMSETUP_VECGEOMCHECKS_H_

#include <cstddef>

namespace o2::simsetup
{

/// Samples random points in the world and compares the volume TGeo locates them in with the
/// volume VecGeom locates them in, which tests the conversion and the level locators on their
/// own, with no Geant4 and no stepping involved. Returns the number of disagreements.
std::size_t checkVecGeomLocation(std::size_t samples);

/// The same comparison, but sampling inside the placements of one named volume rather than over
/// the world. A thin sensitive volume is never sampled often enough by a scan over the world,
/// so this is what tells you whether such a volume is located correctly.
std::size_t checkVecGeomVolume(const char* name, std::size_t perPlacement, std::size_t maxPlacements);

/// Shoots rays from the interaction point and steps them to the world edge with TGeo and with
/// VecGeom, comparing how often each engine reports being in each volume. Stepping is what a
/// containment scan cannot test, and a volume a ray never enters is invisible to a hit count.
void checkVecGeomRays(std::size_t rays);

} // namespace o2::simsetup

#endif
