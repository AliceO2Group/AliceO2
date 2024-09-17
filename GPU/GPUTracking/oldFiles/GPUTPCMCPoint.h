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

/// \file GPUTPCMCPoint.h
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#ifndef GPUTPCMCPOINT_H
#define GPUTPCMCPOINT_H

#include "GPUTPCDef.h"

/**
 * @class GPUTPCMCPoint
 * store MC point information for GPUTPCPerformance
 */
class GPUTPCMCPoint
{
 public:
  GPUTPCMCPoint();

  float X() const { return fX; }
  float Y() const { return fY; }
  float Z() const { return fZ; }
  float Sx() const { return fSx; }
  float Sy() const { return fSy; }
  float Sz() const { return fSz; }
  float Time() const { return fTime; }
  int ISlice() const { return mISlice; }
  int TrackID() const { return fTrackID; }

  void SetX(float v) { fX = v; }
  void SetY(float v) { fY = v; }
  void SetZ(float v) { fZ = v; }
  void SetSx(float v) { fSx = v; }
  void SetSy(float v) { fSy = v; }
  void SetSz(float v) { fSz = v; }
  void SetTime(float v) { fTime = v; }
  void SetISlice(int v) { mISlice = v; }
  void SetTrackID(int v) { fTrackID = v; }

  static bool Compare(const GPUTPCMCPoint& p1, const GPUTPCMCPoint& p2)
  {
    if (p1.fTrackID != p2.fTrackID) {
      return (p1.fTrackID < p2.fTrackID);
    }
    if (p1.mISlice != p2.mISlice) {
      return (p1.mISlice < p2.mISlice);
    }
    return (p1.Sx() < p2.Sx());
  }

  static bool CompareSlice(const GPUTPCMCPoint& p, int slice) { return (p.ISlice() < slice); }

  static bool CompareX(const GPUTPCMCPoint& p, float X) { return (p.Sx() < X); }

 protected:
  float fX;     //* global X position
  float fY;     //* global Y position
  float fZ;     //* global Z position
  float fSx;    //* slice X position
  float fSy;    //* slice Y position
  float fSz;    //* slice Z position
  float fTime;  //* time
  int mISlice;  //* slice number
  int fTrackID; //* mc track number
};

#endif // GPUTPCMCPOINT_H
