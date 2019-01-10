// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   PID.cxx
/// @author Ruben Shahoyan
/// @brief  particle ids, masses, names class implementation

#include "ReconstructionDataFormats/PID.h"
#include <cassert>
#include "FairLogger.h"

using namespace o2::track;

constexpr const char* PID::sNames[NIDs + 1];
constexpr const float PID::sMasses[NIDs];
constexpr const float PID::sMasses2Z[NIDs];
constexpr const int PID::sCharges[NIDs];

//_______________________________
PID::PID(const char* name) : mID(nameToID(name, First))
{
  // construct from the name
  assert(mID < NIDs);
}
