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

#include "ReconstructionDataFormats/Vertex.h"
#include <fmt/printf.h>
#include <iostream>

namespace o2
{
namespace dataformats
{

#ifndef GPUCA_GPUCODE_DEVICE

std::string VertexBase::asString() const
{
  return fmt::format("Vtx {{{:+.4e},{:+.4e},{:+.4e}}} Cov.:{{{{{:.3e}..}},{{{:.3e},{:.3e}..}},{{{:.3e},{:.3e},{:.3e}}}}}",
                     mPos.X(), mPos.Y(), mPos.Z(), mCov[0], mCov[1], mCov[2], mCov[3], mCov[4], mCov[5]);
}

std::ostream& operator<<(std::ostream& os, const o2::dataformats::VertexBase& v)
{
  // stream itself
  os << v.asString();
  return os;
}

void VertexBase::print() const
{
  std::cout << *this << std::endl;
}

#endif

} // namespace dataformats
} // namespace o2
