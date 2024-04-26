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

/// @file   AlignableDetectorTPC.h
/// @author ruben.shahoyan@cern.ch
/// @brief  TPC detector wrapper

#ifndef ALIGNABLEDETECTORTPC_H
#define ALIGNABLEDETECTORTPC_H

#include "Align/AlignableDetector.h"

namespace o2
{
namespace align
{

class AlignableDetectorTPC final : public AlignableDetector
{
 public:
  //
  AlignableDetectorTPC() = default;
  AlignableDetectorTPC(Controller* ctr);
  ~AlignableDetectorTPC() final = default;
  void defineVolumes() final;
  void Print(const Option_t* opt = "") const final;
  //
  int processPoints(GIndex gid, int npntCut, bool inv) final;

  void setTrackTimeStamp(float t) { mTrackTimeStamp = t; }
  float getTrackTimeStamp() const { return mTrackTimeStamp; }

  int getStack(int padrow) const
  {
    for (int i = 0; i < 4; i++) {
      if (padrow <= mStackMinMaxRow[i].second) {
        return i;
      }
    }
    return -1;
  }

  int getDistanceToStackEdge(int padrow) const
  {
    // distance to the stack min or max padrow
    auto st = getStack(padrow);
    if (st < 0) {
      return -999;
    }
    return std::min(padrow - mStackMinMaxRow[st].first, mStackMinMaxRow[st].second - padrow);
  }

 protected:
  //
  float mTrackTimeStamp = 0.f; // use track timestamp in \mus
  static constexpr int NSTACKS = 4;
  const std::array<std::pair<int, int>, NSTACKS> mStackMinMaxRow = {std::pair<int, int>{0, 62}, std::pair<int, int>{63, 96}, std::pair<int, int>{97, 126}, std::pair<int, int>{127, 151}};

  ClassDef(AlignableDetectorTPC, 1);
};
} // namespace align
} // namespace o2
#endif
