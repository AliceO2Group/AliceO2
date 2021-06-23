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

/// \file Hit.cxx
/// \brief Implementation of the Hit class

#include "ITSMFTSimulation/Hit.h"

#include <iostream>
#include <iostream>

ClassImp(o2::itsmft::Hit);

using std::cout;
using std::endl;
using namespace o2::itsmft;
using namespace o2; //::base;

void Hit::Print(const Option_t* opt) const
{
  printf(
    "Det: %5d Track: %6d E.loss: %.3e P: %+.3e %+.3e %+.3e\n"
    "PosIn: %+.3e %+.3e %+.3e PosOut: %+.3e %+.3e %+.3e\n",
    GetDetectorID(), GetTrackID(), GetEnergyLoss(), GetPx(), GetPy(), GetPz(),
    GetStartX(), GetStartY(), GetStartZ(), GetX(), GetY(), GetZ());
}
