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

#include "DataFormatsCalibration/MeanVertexObject.h"

namespace o2
{
namespace dataformats
{

std::string MeanVertexObject::asString() const
{
  return fmt::format("Slopes {{{:+.4e},{:+.4e}}}", mSlopeX, mSlopeY);
}

std::ostream& operator<<(std::ostream& os, const o2::dataformats::MeanVertexObject& o)
{
  // stream itself
  os << o.asString();
  return os;
}

void MeanVertexObject::print() const
{
  VertexBase::print();
  std::cout << *this << std::endl;
}

} // namespace dataformats
} // namespace o2
