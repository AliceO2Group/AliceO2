// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCRow.cxx
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#include "GPUTPCRow.h"
using namespace GPUCA_NAMESPACE::gpu;

#if !defined(GPUCA_GPUCODE)
GPUTPCRow::GPUTPCRow() : mNHits(0), mX(0), mMaxY(0), mGrid(), mHy0(0), mHz0(0), mHstepY(0), mHstepZ(0), mHstepYi(0), mHstepZi(0), mFullSize(0), mHitNumberOffset(0), mFirstHitInBinOffset(0)
{
  // dummy constructor
}

#endif
