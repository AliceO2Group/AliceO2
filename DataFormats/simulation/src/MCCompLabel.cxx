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

#include "SimulationDataFormat/MCCompLabel.h"
#include <iomanip>
#include <ios>
#include <iostream>
#include <cassert>
#include <fmt/format.h>

namespace o2
{

//_____________________________________________
std::ostream& operator<<(std::ostream& os, MCCompLabel const& c)
{
  // stream itself
  if (c.isValid()) {
    os << '[' << c.getSourceID() << '/' << c.getEventID() << '/'
       << (c.isFake() ? '-' : '+') << std::setw(6) << c.getTrackID() << ']';
  } else {
    os << (c.isNoise() ? "[noise]" : "[unset]");
  }
  return os;
}

//_____________________________________________
std::string MCCompLabel::asString() const
{
  // describe itself
  if (isValid()) {
    return fmt::format("[{}/{}{}/{:6d}]", getSourceID(), getEventID(), isFake() ? '-' : '+', getTrackID());
  }
  return isNoise() ? "[noise]" : "[unset]";
}

//_____________________________________________
void MCCompLabel::print() const
{
  // print itself
  std::cout << (MCCompLabel) * this << std::endl;
}

} // namespace o2
