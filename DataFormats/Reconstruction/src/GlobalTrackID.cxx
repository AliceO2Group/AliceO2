// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file  GlobalTrackID.cxx
/// \brief Global index for barrel track: provides provenance (detectors combination), index in respective array and some number of bits
/// \author ruben.shahoyan@cern.ch

#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "Framework/Logger.h"
#include <fmt/printf.h>
#include <iostream>
#include <bitset>

using namespace o2::dataformats;

std::string GlobalTrackID::asString() const
{
  std::bitset<NBitsFlags()> bits{getFlags()};
  return fmt::format("[{:d}/{:d}/{:s}]", getIndex(), getSource(), bits.to_string());
}

std::ostream& o2::dataformats::operator<<(std::ostream& os, const o2::dataformats::GlobalTrackID& v)
{
  // stream itself
  os << v.asString();
  return os;
}

void GlobalTrackID::print() const
{
  LOG(INFO) << asString();
}
