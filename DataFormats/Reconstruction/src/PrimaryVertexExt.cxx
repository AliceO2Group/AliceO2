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

#include "ReconstructionDataFormats/PrimaryVertexExt.h"
#include <fmt/printf.h>
#include <iostream>
#include "CommonUtils/StringUtils.h"

namespace o2
{
namespace dataformats
{

#ifndef GPUCA_ALIGPUCODE
using GTrackID = o2::dataformats::GlobalTrackID;

std::string PrimaryVertexExt::asString() const
{
  auto str = o2::utils::Str::concat_string(PrimaryVertex::asString(), fmt::format("VtxID={} FT0A/C={}/{} FT0T={}", VtxID, FT0A, FT0C, FT0Time));
  for (int i = 0; i < GTrackID::Source::NSources; i++) {
    if (getNSrc(i) > 0) {
      str += fmt::format(" {}={}", GTrackID::getSourceName(i), getNSrc(i));
    }
  }
  return str;
}

std::ostream& operator<<(std::ostream& os, const o2::dataformats::PrimaryVertexExt& v)
{
  // stream itself
  os << v.asString();
  return os;
}

void PrimaryVertexExt::print() const
{
  std::cout << *this << std::endl;
}

#endif

} // namespace dataformats
} // namespace o2
