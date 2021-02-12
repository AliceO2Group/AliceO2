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

#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "Framework/Logger.h"
#include <fmt/printf.h>
#include <iostream>
#include <bitset>
#include <climits>

using namespace o2::dataformats;

std::string VtxTrackRef::asString() const
{
  std::string str = fmt::format("1st entry: {:d} ", getFirstEntry());
  for (int i = 0; i < VtxTrackIndex::NSources; i++) {
    str += fmt::format(", N{:s} : {:d}", VtxTrackIndex::getSourceName(i), getEntriesOfSource(i));
  }
  return str;
}

// set the last +1 element index and finalize all references
void VtxTrackRef::print() const
{
  LOG(INFO) << asString();
}

// set the last +1 element index and check consistency
void VtxTrackRef::setEnd(int end)
{
  if (end <= 0) {
    return; // empty
  }
  setEntries(end - getFirstEntry());
  for (int i = VtxTrackIndex::NSources - 1; i--;) {
    if (getFirstEntryOfSource(i) < 0) {
      throw std::runtime_error(fmt::format("1st entry for source {:d} was not set", i));
    }
    if (getEntriesOfSource(i) < 0) {
      throw std::runtime_error(fmt::format("Source {:d} has negative number of entrie", getEntriesOfSource(i)));
    }
  }
}

std::ostream& o2::dataformats::operator<<(std::ostream& os, const o2::dataformats::VtxTrackRef& v)
{
  // stream itself
  os << v.asString();
  return os;
}
