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

#include "TPCSimulation/Point.h"
#include <iostream>

using std::cout;
using std::endl;

using namespace o2::tpc;

void Point::Print(const Option_t* opt) const
{
  cout << "-I- Point: O2tpc point for track " << GetTrackID()
       << " in detector " << GetDetectorID() << endl;
  cout << "    Position (" << GetX() << ", " << GetY() << ", " << GetZ()
       << ") cm" << endl;
  cout << "    Time " << GetTime() << " ns, n electrons " << GetEnergyLoss() << endl;
}

ClassImp(Point);
ClassImp(HitGroup);
ClassImp(ElementalHit);
