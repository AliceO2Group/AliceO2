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

#ifndef ALIGPU_GPUCODE

std::string PrimaryVertex::asString() const
{
  return o2::utils::concat_string(VertexBase::asString(),
                                  fmt::format("Chi2={:.2f} NCont={:d}: T={:.3f}+-{:.3f} IR=", mChi2, mNContributors, mTimeStamp.getTimeStamp(), mTimeStamp.getTimeStampError()),
                                  mIR.asString());
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
