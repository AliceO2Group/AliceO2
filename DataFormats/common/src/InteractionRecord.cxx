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
#include <fmt/printf.h>

namespace o2
{

#ifndef GPUCA_ALIGPUCODE

std::string InteractionRecord::asString() const
{
  return isDummy() ? std::string{"NotSet"} : fmt::format("BCid: {:4d} Orbit: {:6d}", bc, orbit);
}

std::ostream& operator<<(std::ostream& stream, o2::InteractionRecord const& ir)
{
  stream << ir.asString();
  return stream;
}

std::string InteractionTimeRecord::asString() const
{
  return isDummy() ? InteractionRecord::asString() : InteractionRecord::asString() + fmt::format(" |T in BC(ns): {:.3f}", timeInBCNS);
}

std::ostream& operator<<(std::ostream& stream, o2::InteractionTimeRecord const& ir)
{
  stream << ir.asString();
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

#endif

} // namespace o2
