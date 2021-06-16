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

/// \file GPUTPCHit.h
/// \author Sergey Gorbunov, David Rohr

#ifndef GPUTPCHIT_H
#define GPUTPCHIT_H

#include "GPUTPCDef.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
/**
 * @class GPUTPCHit
 *
 * The GPUTPCHit class is the internal representation
 * of the TPC clusters for the GPUTPCTracker algorithm.
 *
 */
class GPUTPCHit
{
 public:
  GPUhd() float Y() const { return mY; }
  GPUhd() float Z() const { return mZ; }

  GPUhd() void SetY(float v) { mY = v; }
  GPUhd() void SetZ(float v) { mZ = v; }

 protected:
  float mY, mZ; // Y and Z position of the TPC cluster

 private:
  friend class GPUTPCNeighboursFinder;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTPCHIT_H
