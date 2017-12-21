// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DetID.cxx
/// @author Ruben Shahoyan
/// @brief  detector ids, masks, names class implementation

#include "DetectorsBase/DetID.h"
#include <cassert>
#include "FairLogger.h"

using namespace o2::Base;

ClassImp(o2::Base::DetID);

constexpr const char* DetID::sDetNames[DetID::nDetectors + 1];
constexpr std::array<DetID::mask_t, DetID::nDetectors> DetID::sMasks;

//_______________________________
DetID::DetID(const char* name) : mID(nameToID(name, First))
{
  // construct from the name
  assert(mID < nDetectors);
}
