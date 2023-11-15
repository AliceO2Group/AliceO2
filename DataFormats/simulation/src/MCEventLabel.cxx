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

#include "SimulationDataFormat/MCEventLabel.h"
#include <iomanip>
#include <ios>
#include <iostream>
#include <cassert>
#include <fmt/format.h>

using namespace o2;

//_____________________________________________
void MCEventLabel::print() const
{
  // print itself
  std::cout << (MCEventLabel) * this << std::endl;
}

//_____________________________________________
std::string MCEventLabel::asString() const
{
  // stream itself
  return isSet() ? fmt::format("[{}/{}/{}]", getSourceID(), getEventID(), getCorrWeight()) : "[unset]";
}

//_____________________________________________
std::ostream& operator<<(std::ostream& os, const o2::MCEventLabel& c)
{
  // stream itself
  if (c.isSet()) {
    os << '[' << c.getSourceID() << '/' << c.getEventID() << '/' << c.getCorrWeight() << ']';
  } else {
    os << "[unset]";
  }
  return os;
}
