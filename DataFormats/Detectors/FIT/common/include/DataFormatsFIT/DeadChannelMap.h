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

/// \file DeadChannelMap.h
/// \brief Dead channel map for FIT
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#ifndef O2_FIT_DEADCHANNELMAP_H
#define O2_FIT_DEADCHANNELMAP_H

#include <cstdint>
#include <unordered_map>

namespace o2
{
namespace fit
{

struct DeadChannelMap {
  /// Dead channel map as 'channel id - state' pairs. true = alive, false = dead.
  std::unordered_map<uint8_t, bool> map;

  void setChannelAlive(const uint8_t& chId, const bool isAlive)
  {
    map[chId] = isAlive;
  }

  const bool isChannelAlive(const uint8_t& chId) const
  {
    return map.at(chId);
  }

  void clear()
  {
    map.clear();
  }

  ClassDefNV(DeadChannelMap, 1);
};

} // namespace fit
} // namespace o2

#endif // O2_FIT_DEADCHANNELMAP_H