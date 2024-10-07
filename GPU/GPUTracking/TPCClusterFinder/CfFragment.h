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

/// \file CfFragment.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_CF_FRAGMENT_H
#define O2_GPU_CF_FRAGMENT_H

#include "clusterFinderDefs.h"
#include "GPUCommonMath.h"

namespace GPUCA_NAMESPACE::gpu
{

struct CfFragment {

  enum : tpccf::TPCTime {
    OverlapTimebins = 8,
  };

  // Time offset of this sub slice within the entire time slice
  tpccf::TPCTime start = 0;
  // Number of time bins to process in this slice
  tpccf::TPCFragmentTime length = 0;

  size_t digitsStart = 0; // Start digits in this fragment. Only used when zero suppression is skipped

  uint32_t index = 0;

  bool hasBacklog = false;
  bool hasFuture = false;
  tpccf::TPCTime totalSliceLength = 0;
  tpccf::TPCFragmentTime maxSubSliceLength = 0;

  GPUdDefault() CfFragment() CON_DEFAULT;

  GPUd() CfFragment(tpccf::TPCTime totalSliceLen, tpccf::TPCFragmentTime maxSubSliceLen) : CfFragment(0, false, 0, totalSliceLen, maxSubSliceLen) {}

  GPUdi() bool isEnd() const { return length == 0; }

  GPUdi() CfFragment next() const
  {
    return CfFragment{index + 1, hasFuture, tpccf::TPCTime(start + length - (hasFuture ? 2 * OverlapTimebins : 0)), totalSliceLength, maxSubSliceLength};
  }

  GPUdi() uint32_t count() const
  {
    return (totalSliceLength + maxSubSliceLength - 4 * OverlapTimebins - 1) / (maxSubSliceLength - 2 * OverlapTimebins);
  }

  GPUdi() tpccf::TPCTime first() const
  {
    return start;
  }

  GPUdi() tpccf::TPCTime last() const
  {
    return start + length;
  }

  GPUdi() bool contains(tpccf::TPCTime t) const
  {
    return first() <= t && t < last();
  }

  // Wether a timebin falls into backlog or future
  GPUdi() bool isOverlap(tpccf::TPCFragmentTime t) const
  {
    return (hasBacklog ? t < OverlapTimebins : false) || (hasFuture ? t >= (length - OverlapTimebins) : false);
  }

  GPUdi() tpccf::TPCFragmentTime lengthWithoutOverlap() const
  {
    return length - (hasBacklog ? OverlapTimebins : 0) - (hasFuture ? OverlapTimebins : 0);
  }

  GPUdi() tpccf::TPCFragmentTime firstNonOverlapTimeBin() const
  {
    return (hasBacklog ? OverlapTimebins : 0);
  }

  GPUdi() tpccf::TPCFragmentTime lastNonOverlapTimeBin() const
  {
    return length - (hasFuture ? OverlapTimebins : 0);
  }

  GPUdi() tpccf::TPCFragmentTime toLocal(tpccf::TPCTime t) const
  {
    return t - first();
  }

  GPUdi() tpccf::TPCTime toGlobal(tpccf::TPCFragmentTime t) const
  {
    return t + first();
  }

 private:
  GPUd() CfFragment(uint32_t index_, bool hasBacklog_, tpccf::TPCTime start_, tpccf::TPCTime totalSliceLen, tpccf::TPCFragmentTime maxSubSliceLen)
  {
    this->index = index_;
    this->hasBacklog = hasBacklog_;
    this->start = start_;
    tpccf::TPCTime remainder = totalSliceLen - start;
    this->hasFuture = remainder > tpccf::TPCTime(maxSubSliceLen);
    this->length = hasFuture ? maxSubSliceLen : remainder;
    this->totalSliceLength = totalSliceLen;
    this->maxSubSliceLength = maxSubSliceLen;
  }
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
