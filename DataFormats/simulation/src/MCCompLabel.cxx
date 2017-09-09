// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "SimulationDataFormat/MCCompLabel.h"

using namespace o2;

ClassImp(o2::MCCompLabel);

std::ostream& operator<<(std::ostream& os, const o2::MCCompLabel& c)
{
  // stream itself
  if (c.isSet()) {
    os << '[' << c.getSourceID() << '/' << c.getEventID() << '/'
       << std::setw(6) << c.getTrackID() << ']';
  } else {
    os << "[unset]";
  }
  return os;
}
