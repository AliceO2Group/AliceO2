// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ReconstructionDataFormats/PrimaryVertex.h"
#include <fmt/printf.h>
#include <iostream>
#include "CommonUtils/StringUtils.h"

namespace o2
{
namespace dataformats
{

#ifndef GPUCA_ALIGPUCODE

std::string PrimaryVertex::asString() const
{
  auto str = o2::utils::Str::concat_string(VertexBase::asString(),
                                           fmt::format("Chi2={:.2f} NCont={:d}: T={:.3f}+-{:.3f} IR=", mChi2, mNContributors, mTimeStamp.getTimeStamp(), mTimeStamp.getTimeStampError()),
                                           mIRMin.asString());
  if (!hasUniqueIR()) {
    str = o2::utils::Str::concat_string(str, " : ", mIRMax.asString());
  }
  return str;
}

std::ostream& operator<<(std::ostream& os, const o2::dataformats::PrimaryVertex& v)
{
  // stream itself
  os << v.asString();
  return os;
}

void PrimaryVertex::print() const
{
  std::cout << *this << std::endl;
}

#endif

} // namespace dataformats
} // namespace o2
