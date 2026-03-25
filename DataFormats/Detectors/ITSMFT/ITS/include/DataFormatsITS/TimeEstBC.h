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

#ifndef O2_TRACKINGITS_TIMEESTBC_H_
#define O2_TRACKINGITS_TIMEESTBC_H_

#include <limits>
#include <cstdint>
#include "CommonDataFormat/TimeStamp.h"
#include "GPUCommonRtypes.h"
#include "GPUCommonDef.h"
#include "GPUCommonMath.h"

namespace o2::its
{
// Time estimates are given in BC
// error needs to cover maximum 1 orbit
using TimeStampType = uint32_t;
using TimeStampErrorType = uint16_t;
// this is an symmetric time error [t0-tE, t0+tE]
using TimeStamp = o2::dataformats::TimeStampWithError<float, float>;
// this is an asymmetric time interval [t0, t0+tE] used for internal calculations
class TimeEstBC : public o2::dataformats::TimeStampWithError<TimeStampType, TimeStampErrorType>
{
  using Base = o2::dataformats::TimeStampWithError<TimeStampType, TimeStampErrorType>;

 public:
  GPUhdDefault() TimeEstBC() = default;
  GPUhdi() TimeEstBC(TimeStampType t, TimeStampErrorType e) : Base(t, e) {}

  // convert to symmetric center+-half representation
  GPUhdi() its::TimeStamp makeSymmetrical() const noexcept
  {
    const auto start = static_cast<float>(this->getTimeStamp());
    const float half = (float)this->getTimeStampError() / 2.f;
    return {start + half, half};
  }

  // check if timestamps overlap within their interval
  GPUhdi() bool isCompatible(const TimeEstBC& o) const noexcept
  {
    return this->upper() > o.lower() && o.upper() > this->lower();
  }

  // check if this time interval is fully contained within o
  GPUhdi() bool isContainedIn(const TimeEstBC& o) const noexcept
  {
    return this->lower() >= o.lower() && this->upper() <= o.upper();
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

  // upper bound of interval t0+tE
  GPUhdi() TimeStampType upper() const noexcept
  {
    TimeStampType t = this->getTimeStamp();
    TimeStampType e = this->getTimeStampError();
    constexpr TimeStampType max = std::numeric_limits<TimeStampType>::max();
    return (t > (max - e)) ? max : t + e;
  }

  // lower bound of interval t0
  GPUhdi() TimeStampType lower() const noexcept
  {
    return this->getTimeStamp();
  }

 private:
  // intersect with the other timestamp
  // this assumes already that both overlap
  GPUhdi() void add(const TimeEstBC& o) noexcept
  {
    const TimeStampType lo = o2::gpu::CAMath::Max(this->lower(), o.lower());
    const TimeStampType hi = o2::gpu::CAMath::Min(this->upper(), o.upper());
    this->setTimeStamp(lo);
    this->setTimeStampError(static_cast<TimeStampErrorType>(hi - lo));
  }

  ClassDefNV(TimeEstBC, 1);
};

} // namespace o2::its

#endif
