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
using TimeStampType = uint32_t;
using TimeStampErrorType = uint16_t;
class TimeEstBC : public o2::dataformats::TimeStampWithError<TimeStampType, TimeStampErrorType>
{
 public:
  using Base = o2::dataformats::TimeStampWithError<TimeStampType, TimeStampErrorType>;
  GPUhdDefault() TimeEstBC() = default;
  GPUhdi() TimeEstBC(TimeStampType t, TimeStampErrorType e) : Base(t, e) {}

  // check if timestamps overlap within their interval
  GPUhdi() bool isCompatible(const TimeEstBC& o) const noexcept
  {
    return !(upper() < o.lower() || o.upper() < lower());
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

 private:
  // add the other timestmap to this one
  // this assumes already that both overlap
  GPUhdi() void add(const TimeEstBC& o) noexcept
  {
    const TimeStampType lo = o2::gpu::CAMath::Max(lower(), o.lower());
    const TimeStampType hi = o2::gpu::CAMath::Min(upper(), o.upper());
    const TimeStampType half = (hi - lo) / 2u;
    this->setTimeStamp(lo + half);
    this->setTimeStampError(static_cast<TimeStampErrorType>(half));
  }

  GPUhdi() TimeStampType upper() const noexcept
  {
    TimeStampType t = this->getTimeStamp();
    TimeStampType e = this->getTimeStampError();
    constexpr TimeStampType max = std::numeric_limits<TimeStampType>::max();
    return (t > (max - e)) ? max : t + e;
  }

  GPUhdi() TimeStampType lower() const noexcept
  {
    TimeStampType t = this->getTimeStamp();
    TimeStampType e = this->getTimeStampError();
    return (t > e) ? (t - e) : 0u;
  }

  ClassDefNV(TimeEstBC, 1);
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