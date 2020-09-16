// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file  VtxTrackIndex.h
/// \brief Index of track attached to vertx: index in its proper container, container source and flags
/// \author ruben.shahoyan@cern.ch

#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "Framework/Logger.h"
#include <fmt/printf.h>
#include <iostream>
#include <bitset>

using namespace o2::dataformats;

std::string VtxTrackIndex::asString() const
{
  std::bitset<NBitsFlags()> bits{getFlags()};
  return fmt::format("[{:d}/{:d}/{:s}]", getIndex(), getSource(), bits.to_string());
}

std::ostream& o2::dataformats::operator<<(std::ostream& os, const o2::dataformats::VtxTrackIndex& v)
{
  // stream itself
  os << v.asString();
  return os;
}

void VtxTrackIndex::print() const
{
  LOG(INFO) << asString();
}
