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

#ifndef O2_SIMSETUP_VECGEOMNAVIGATION_H_
#define O2_SIMSETUP_VECGEOMNAVIGATION_H_

namespace o2::simsetup
{

/// Whether this build of O2 has the VecGeom navigation backend, i.e. whether TGeo2VecGeom and a
/// VecGeom with BVHNavigatorV were found when O2 was configured.
bool isVecGeomNavigationAvailable();

/// Replaces Geant4's tracking navigator by one that answers every navigation query from
/// VecGeom. The Geant4 geometry, its materials and the touchable the scoring code reads stay
/// the ones g4root built from TGeo, so only navigation changes.
///
/// Call after the TGeant4 engine has been constructed: the Geant4 hierarchy this needs to map
/// onto is built while TG4RunManager configures itself. Aborts if the backend is missing.
void installVecGeomNavigator();

} // namespace o2::simsetup

#endif
