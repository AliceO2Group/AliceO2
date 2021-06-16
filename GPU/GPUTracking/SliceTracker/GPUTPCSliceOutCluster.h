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

/// \file GPUTPCSliceOutCluster.h
/// \author Sergey Gorbunov, David Rohr

#ifndef GPUTPCSLICEOUTCLUSTER_H
#define GPUTPCSLICEOUTCLUSTER_H

#include "GPUTPCDef.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
/**
 * @class GPUTPCSliceOutCluster
 * GPUTPCSliceOutCluster class contains clusters which are assigned to slice tracks.
 * It is used to send the data from TPC slice trackers to the GlobalMerger
 */
class GPUTPCSliceOutCluster
{
 public:
  GPUhd() void Set(unsigned int id, unsigned char row, unsigned char flags, unsigned short amp, float x, float y, float z)
  {
    mRow = row;
    mFlags = flags;
    mId = id;
    mAmp = amp;
    mX = x;
    mY = y;
    mZ = z;
  }

  GPUhd() float GetX() const { return mX; }
  GPUhd() float GetY() const { return mY; }
  GPUhd() float GetZ() const { return mZ; }
  GPUhd() unsigned short GetAmp() const { return mAmp; }
  GPUhd() unsigned int GetId() const { return mId; }
  GPUhd() unsigned char GetRow() const { return mRow; }
  GPUhd() unsigned char GetFlags() const { return mFlags; }

 private:
  unsigned int mId;     // Id
  unsigned char mRow;   // row
  unsigned char mFlags; // flags
  unsigned short mAmp;  // amplitude
  float mX;             // coordinates
  float mY;             // coordinates
  float mZ;             // coordinates

#ifdef GPUCA_TPC_RAW_PROPAGATE_PAD_ROW_TIME
 public:
  float mPad;
  float mTime;
#endif
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
