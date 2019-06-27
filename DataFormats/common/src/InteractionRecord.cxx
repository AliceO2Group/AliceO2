// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "CommonDataFormat/InteractionRecord.h"
#include <iostream>

namespace o2
{

//const double InteractionRecord::DummyTime = InteractionRecord::bc2ns(0xffff,0xffffffff);

std::ostream& operator<<(std::ostream& stream, o2::InteractionRecord const& ir)
{
  stream << "BCid: " << ir.bc << " Orbit: " << ir.orbit;
  return stream;
}

std::ostream& operator<<(std::ostream& stream, o2::InteractionTimeRecord const& ir)
{
  stream << "BCid: " << ir.bc << " Orbit: " << ir.orbit << " T(ns): " << ir.timeNS;
  return stream;
}

void InteractionRecord::print() const
{
  std::cout << (*this) << std::endl;
}

void InteractionTimeRecord::print() const
{
  std::cout << (*this) << std::endl;
}

} // namespace o2
