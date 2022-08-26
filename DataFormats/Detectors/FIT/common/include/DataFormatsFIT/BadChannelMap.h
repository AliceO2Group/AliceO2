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

/// \file BadChannelMap.h
/// \brief Bad channel map for FIT
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#ifndef O2_FIT_BADCHANNELMAP_H
#define O2_FIT_BADCHANNELMAP_H

#include <cstdint>
#include <unordered_map>

namespace o2
{
namespace fit
{

struct BadChannelMap {
  /// Bad channel map as 'channel id - state' pairs. true = good, false = bad.
  std::unordered_map<uint8_t, bool> map;

  void setChannelGood(const uint8_t& chId, const bool isGood)
  {
    map[chId] = isGood;
  }

  const bool isChannelGood(const uint8_t& chId) const
  {
    return map.at(chId);
  }

  void clear()
  {
    map.clear();
  }

  ClassDefNV(BadChannelMap, 1);
};

} // namespace fit
} // namespace o2

#endif // O2_FIT_BADCHANNELMAP_H