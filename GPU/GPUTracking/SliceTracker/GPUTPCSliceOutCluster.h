// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCSliceOutCluster.h
/// \author Sergey Gorbunov, David Rohr

#ifndef GPUTPCSLICEOUTCLUSTER_H
#define GPUTPCSLICEOUTCLUSTER_H

#include "GPUTPCDef.h"

namespace o2
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
    mAmp = amp;
    mId = id;
    mX = x;
    mY = y;
    mZ = z;
  }

  GPUhd() float GetX() const { return mX; }
  GPUhd() float GetY() const { return mY; }
  GPUhd() float GetZ() const { return mZ; }
  GPUhd() unsigned int GetId() const { return mId; }
  GPUhd() unsigned char GetRow() const { return mRow; }
  GPUhd() unsigned char GetFlags() const { return mFlags; }
  GPUhd() unsigned short GetAmp() const { return mAmp; }

 private:
  unsigned int mId;     // Id ( slice, patch, cluster )
  unsigned char mRow;   // row
  unsigned char mFlags; // flags
  unsigned short mAmp;  // amplitude
  float mX;             // coordinates
  float mY;             // coordinates
  float mZ;             // coordinates

#ifdef GMPropagatePadRowTime
 public:
  float mPad;
  float mTime;
#endif
};
}
} // namespace o2::gpu

#endif
