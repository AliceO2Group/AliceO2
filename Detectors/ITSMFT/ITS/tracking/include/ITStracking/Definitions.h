// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file Definitions.h
/// \brief

#ifndef TRACKINGITS_DEFINITIONS_H_
#define TRACKINGITS_DEFINITIONS_H_

#include <type_traits>
#include <cstdint>

namespace o2::its
{

enum class TrackletMode {
  Layer0Layer1 = 0,
  Layer1Layer2 = 2
};

template <bool IsConst, typename T>
using maybe_const = typename std::conditional<IsConst, const T, T>::type;

// simple implemnetion of logging with exp. backoff
struct LogLogThrottler {
  uint64_t evCount{0};
  uint64_t nextLog{1};
  int32_t iteration{-1};
  int32_t layer{-1};
  bool needToLog(int32_t iter, int32_t lay)
  {
    if (iteration != iter || layer != lay) {
      iteration = iter;
      layer = lay;
      evCount = 0;
      nextLog = 1;
    }
    if (++evCount > nextLog) {
      nextLog *= 2;
      return true;
    }
    return false;
  }
};

struct TimingStats {
  std::uint64_t calls = 0;
  double totalTimeMs = 0.;

  void add(double timeMs)
  {
    ++calls;
    totalTimeMs += timeMs;
  }
  double averageTimeMs() const { return calls ? totalTimeMs / static_cast<double>(calls) : 0.; }
};

} // namespace o2::its

#endif
