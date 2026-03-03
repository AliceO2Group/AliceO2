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

#include <limits>
#include <type_traits>
#include <cstdint>

#include "SimulationDataFormat/MCCompLabel.h"
#include "CommonDataFormat/TimeStamp.h"
#include "ReconstructionDataFormats/Vertex.h"
#include "GPUCommonRtypes.h"
#include "GPUCommonDef.h"

namespace o2::its
{

enum class TrackletMode {
  Layer0Layer1 = 0,
  Layer1Layer2 = 2
};

template <bool IsConst, typename T>
using maybe_const = typename std::conditional<IsConst, const T, T>::type;

// Time estimates are given in BC
// error needs to cover maximum 1 orbit
// this is an inclusive symmetric time error [t0-tE, t0+tE]
struct TimeEstBC : public o2::dataformats::TimeStampWithError<uint32_t, uint16_t> {
  using Base = o2::dataformats::TimeStampWithError<uint32_t, uint16_t>;
  GPUhdDefault() TimeEstBC() = default;
  GPUhdi() TimeEstBC(uint32_t t, uint16_t e) : Base(t, e) {}

  // check if timestamps overlap within their interval
  GPUhdi() bool isCompatible(const TimeEstBC& o) const noexcept
  {
    return !(upper() < o.lower() || o.upper() < lower());
  }

  // add the other timestmap to this one
  // this assumes already that both overlap
  GPUhdi() void add(const TimeEstBC& o) noexcept
  {
    const uint32_t lo = o2::gpu::CAMath::Max(lower(), o.lower());
    const uint32_t hi = o2::gpu::CAMath::Min(upper(), o.upper());
    const uint32_t half = (hi - lo) / 2u;
    this->setTimeStamp(lo + half);
    this->setTimeStampError(static_cast<uint16_t>(half));
  }

  GPUhdi() TimeEstBC& operator+=(const TimeEstBC& o) noexcept
  {
    add(o);
    return *this;
  }

  GPUhdi() TimeEstBC operator+(const TimeEstBC& o) const noexcept
  {
    TimeEstBC res = *this;
    res += o;
    return res;
  }

  GPUhdi() uint32_t lower() const noexcept
  {
    uint32_t t = this->getTimeStamp();
    uint32_t e = this->getTimeStampError();
    return (t > e) ? (t - e) : 0u;
  }
  GPUhdi() uint32_t upper() const noexcept
  {
    uint32_t t = this->getTimeStamp();
    uint32_t e = this->getTimeStampError();
    constexpr uint32_t max = std::numeric_limits<uint32_t>::max();
    return (t > (max - e)) ? max : t + e;
  }

  ClassDef(TimeEstBC, 1);
};
using Vertex = o2::dataformats::Vertex<TimeEstBC>;
using VertexLabel = std::pair<o2::MCCompLabel, float>;

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
} // namespace o2::its

#endif
