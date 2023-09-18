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

/// \file GPUTriggerOutputs.h
/// \author David Rohr

#ifndef GPUTRIGGEROUTPUTS_H
#define GPUTRIGGEROUTPUTS_H

#include "GPUCommonDef.h"
#include <unordered_set>
#include <array>
#ifdef GPUCA_HAVE_O2HEADERS
#include "DataFormatsTPC/ZeroSuppression.h"
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{

struct GPUTriggerOutputs {
#ifdef GPUCA_HAVE_O2HEADERS
  struct hasher {
    size_t operator()(const std::array<unsigned long, o2::tpc::TPCZSHDRV2::TRIGGER_WORD_SIZE / sizeof(unsigned long)>& key) const
    {
      std::hash<unsigned long> std_hasher;
      size_t result = 0;
      for (size_t i = 0; i < key.size(); ++i) {
        result ^= std_hasher(key[i]);
      }
      return result;
    }
  };

  std::unordered_set<std::array<unsigned long, o2::tpc::TPCZSHDRV2::TRIGGER_WORD_SIZE / sizeof(unsigned long)>, hasher> triggers;
  static_assert(o2::tpc::TPCZSHDRV2::TRIGGER_WORD_SIZE % sizeof(unsigned long) == 0);
#endif
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
