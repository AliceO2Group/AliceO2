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
#include <iomanip>
#include <ios>
#include <iostream>
#include <cassert>

using namespace o2;

ClassImp(o2::MCCompLabel);

//_____________________________________________
void MCCompLabel::print() const
{
  // print itself
  std::cout << (MCCompLabel)*this << std::endl;
}

//_____________________________________________
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

 //_____________________________________________
void MCCompLabel::checkFieldConsistensy()
{
  // check if the fields are defined consistently
  static_assert(nbitsTrackID==sizeof(int)*8, "TrackID must have int size");
  static_assert(nbitsTrackID+nbitsEvID+nbitsSrcID<=sizeof(ULong64_t)*8,
                "Fields cannot be stored in 64 bits");
}
