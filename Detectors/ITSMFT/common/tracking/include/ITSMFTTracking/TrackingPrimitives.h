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

#ifndef ALICEO2_ITSMFT_TRACKING_TRACKINGPRIMITIVES_H_
#define ALICEO2_ITSMFT_TRACKING_TRACKINGPRIMITIVES_H_

#include "DataFormatsITS/TimeEstBC.h"
#include "GPUCommonDef.h"
#include "ITSMFTTracking/Constants.h"
#include "ITSMFTTracking/MathUtils.h"

#include <type_traits>

namespace o2::itsmft::tracking
{

// Per-iteration connection between two sorted measurement locators.
struct Tracklet {
  GPUhdDefault() Tracklet() = default;
  GPUhd() Tracklet(int first, int second, float tanL, float azimuth, const o2::its::TimeEstBC& time)
    : firstClusterIndex{first}, secondClusterIndex{second}, tanLambda{tanL}, phi{azimuth}, mTime{time}
  {
  }

  GPUhd() bool operator<(const Tracklet& other) const noexcept
  {
    return firstClusterIndex != other.firstClusterIndex ? firstClusterIndex < other.firstClusterIndex
                                                        : secondClusterIndex < other.secondClusterIndex;
  }
  GPUhd() bool operator==(const Tracklet& other) const noexcept
  {
    return firstClusterIndex == other.firstClusterIndex && secondClusterIndex == other.secondClusterIndex;
  }
  GPUhd() bool isCompatible(const Tracklet& other) const { return mTime.isCompatible(other.mTime); }
  GPUhd() auto& getTimeStamp() noexcept { return mTime; }
  GPUhd() const auto& getTimeStamp() const noexcept { return mTime; }

  int firstClusterIndex{o2::its::constants::UnusedIndex};
  int secondClusterIndex{o2::its::constants::UnusedIndex};
  float tanLambda{o2::its::constants::UnsetValue};
  float phi{o2::its::constants::UnsetValue};
  o2::its::TimeEstBC mTime;
};

static_assert(std::is_trivially_copyable_v<Tracklet>);

} // namespace o2::itsmft::tracking

#endif // ALICEO2_ITSMFT_TRACKING_TRACKINGPRIMITIVES_H_
