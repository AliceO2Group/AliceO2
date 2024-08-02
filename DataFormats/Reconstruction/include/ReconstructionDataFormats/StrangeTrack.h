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

/// \file StrangeTrack.h
/// \brief
///

#ifndef _ALICEO2_STRANGETRACK_
#define _ALICEO2_STRANGETRACK_

#include <array>
#include "ReconstructionDataFormats/Track.h"

namespace o2
{
namespace dataformats
{

enum kPartType { kStrkV0,
                 kStrkCascade,
                 kStrkThreeBody };

struct StrangeTrack {
  kPartType mPartType;
  o2::track::TrackParCovF mMother;
  unsigned int mITSRef = -1;
  unsigned int mDecayRef = -1;
  std::array<float, 3> mDecayVtx;
  std::array<float, 3> mDecayMom;
  std::array<float, 2> mMasses;    // V0: hypertriton and hyperhydrongen4, cascade: Xi and Omega.
  unsigned int mClusterSizes = 0u; // same encoding used for the ITS track
  float mMatchChi2;
  float mTopoChi2;

  void setClusterSize(int l, int size)
  {
    if (l >= 8) {
      return;
    }
    if (size > 15) {
      size = 15;
    }
    mClusterSizes &= ~(0xf << (l * 4));
    mClusterSizes |= (size << (l * 4));
  }

  int getClusterSize(int l) const
  {
    if (l >= 7) {
      return 0;
    }
    return (mClusterSizes >> (l * 4)) & 0xf;
  }

  int getClusterSizes() const
  {
    return mClusterSizes;
  }

  float getAverageClusterSize() const
  {
    int nClusters = 0;
    int nClustersSize = 0;
    for (int i = 0; i < 7; ++i) {
      int size = getClusterSize(i);
      if (size > 0) {
        nClustersSize += size;
        nClusters++;
      }
    }
    return nClusters > 0 ? static_cast<float>(nClustersSize) / nClusters : 0.f;
  }
};

} // namespace dataformats
} // namespace o2

#endif // _ALICEO2_STRANGETRACK_
