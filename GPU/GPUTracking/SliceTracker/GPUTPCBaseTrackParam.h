// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCBaseTrackParam.h
/// \author David Rohr, Sergey Gorbunov

#ifndef GPUTPCBASETRACKPARAM_H
#define GPUTPCBASETRACKPARAM_H

#include "GPUTPCDef.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
MEM_CLASS_PRE()
class GPUTPCTrackParam;

/**
 * @class GPUTPCBaseTrackParam
 *
 * GPUTPCBaseTrackParam class contains track parameters
 * used in output of the GPUTPCTracker slice tracker.
 * This class is used for transfer between tracker and merger and does not contain the covariance matrice
 */
MEM_CLASS_PRE()
class GPUTPCBaseTrackParam
{
 public:
  GPUd() float X() const { return mX; }
  GPUd() float Y() const { return mP[0]; }
  GPUd() float Z() const { return mP[1]; }
  GPUd() float SinPhi() const { return mP[2]; }
  GPUd() float DzDs() const { return mP[3]; }
  GPUd() float QPt() const { return mP[4]; }
  GPUd() float ZOffset() const { return mZOffset; }

  GPUhd() float GetX() const { return mX; }
  GPUhd() float GetY() const { return mP[0]; }
  GPUhd() float GetZ() const { return mP[1]; }
  GPUhd() float GetSinPhi() const { return mP[2]; }
  GPUhd() float GetDzDs() const { return mP[3]; }
  GPUhd() float GetQPt() const { return mP[4]; }
  GPUhd() float GetZOffset() const { return mZOffset; }

  GPUd() float GetKappa(float Bz) const { return -mP[4] * Bz; }

  GPUhd() MakeType(const float*) Par() const { return mP; }
  GPUd() const MakeType(float*) GetPar() const { return mP; }
  GPUd() float GetPar(int i) const { return (mP[i]); }

  GPUhd() void SetPar(int i, float v) { mP[i] = v; }

  GPUd() void SetX(float v) { mX = v; }
  GPUd() void SetY(float v) { mP[0] = v; }
  GPUd() void SetZ(float v) { mP[1] = v; }
  GPUd() void SetSinPhi(float v) { mP[2] = v; }
  GPUd() void SetDzDs(float v) { mP[3] = v; }
  GPUd() void SetQPt(float v) { mP[4] = v; }
  GPUd() void SetZOffset(float v) { mZOffset = v; }

 private:
  // WARNING, Track Param Data is copied in the GPU Tracklet Constructor element by element instead of using copy constructor!!!
  // This is neccessary for performance reasons!!!
  // Changes to Elements of this class therefore must also be applied to TrackletConstructor!!!
  float mX; // x position
  float mZOffset;
  float mP[5]; // 'active' track parameters: Y, Z, SinPhi, DzDs, q/Pt
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
