// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "SimulationDataFormat/RunContext.h"
#include <iostream>

using namespace o2::steer;

void RunContext::printCollisionSummary() const
{
  std::cout << "Summary of RunContext --\n";
  std::cout << "Number of Collisions " << mEventRecords.size() << "\n";
  for (int i = 0; i < mEventRecords.size(); ++i) {
    std::cout << "Collision " << i << " TIME " << mEventRecords[i].timeNS;
    for (auto& e : mEventParts[i]) {
      std::cout << " (" << e.sourceID << " , " << e.entryID << ")";
    }
    std::cout << "\n";
  }
}
