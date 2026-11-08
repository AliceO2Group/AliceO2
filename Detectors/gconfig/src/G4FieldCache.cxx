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

#include "SimSetup/G4FieldCache.h"

#include <G4Field.hh>
#include <G4FieldManager.hh>
#include <G4FieldManagerStore.hh>

namespace o2::simsetup
{

void invalidateFieldCache()
{
  // the field caches reuse their value only near the last point, so one query far away invalidates them
  constexpr G4double farAway[4] = {1.e12, 1.e12, 1.e12, 0.};
  G4double value[6];
  // the field manager store holds the field managers of the calling thread
  for (auto manager : *G4FieldManagerStore::GetInstance()) {
    if (auto field = manager->GetDetectorField()) {
      field->GetFieldValue(farAway, value);
    }
  }
}

} // namespace o2::simsetup
