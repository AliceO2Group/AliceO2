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

/// @file   AlignmentPoint.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Meausered point in the sensor.

/**
 * Compact alignment point info for debugging
 */

#ifndef ALGPNTDBG_H
#define ALGPNTDBG_H

#include "Align/AlignmentPoint.h"

namespace o2
{
namespace align
{

struct AlgPntDbg {
 public:
  using DetID = o2::detectors::DetID;
  //
  enum {
    UpperLeg = 0
  };
  //
  AlgPntDbg() = default;
  AlgPntDbg(const AlgPntDbg&) = default;
  ~AlgPntDbg() = default;
  AlgPntDbg& operator=(const AlgPntDbg& other) = default;
  AlgPntDbg(const AlignmentPoint* point) : mDetID(point->getDetID()), mSID(point->getSID()), mAlpha(point->getAlphaSens()), mX(point->getXTracking()), mY(point->getYTracking()), mZ(point->getZTracking()), mErrYY(point->getYZErrTracking()[0]), mErrZZ(point->getYZErrTracking()[2]), mErrYZ(point->getYZErrTracking()[1])
  {
    mSinAlp = std::sin(mAlpha);
    mCosAlp = std::cos(mAlpha);
    mSnp = point->getTrParamWSA()[2]; // track Snp at the sensor
    if (point->isInvDir()) {
      setUpperLeg();
    }
  }

  float getR() const { return std::sqrt(mX * mX + mY * mY); }
  float getYTrack() const { return mY + mYRes; }
  float getZTrack() const { return mZ + mZRes; }
  float getXTrack() const { return mX; }
  float getXLab() const { return mX * mCosAlp - mY * mSinAlp; }
  float getYLab() const { return mX * mSinAlp + mY * mCosAlp; }
  float getZLap() const { return mZ; }
  float getXTrackLab() const { return mX * mCosAlp - getYTrack() * mSinAlp; }
  float getYTrackLab() const { return mX * mSinAlp + getYTrack() * mCosAlp; }
  float getZTrackLab() const { return getZTrack(); }
  float getPhi() const { return std::atan2(getYLab(), getXLab()); }
  void setFlag(int i) { mFlags |= 0x1 << i; }
  bool getFlag(int i) const { return (mFlags & (0x1 << i)) != 0; }

  void setUpperLeg() { setFlag(int(UpperLeg)); }
  bool isUpperLeg() const { return getFlag(int(UpperLeg)); }

  int mDetID{};        // DetectorID
  int16_t mSID = -1;   // sensor ID in the detector
  uint16_t mFlags = 0; // flags
  float mAlpha = 0.f;  // Alpha of tracking frame
  float mSinAlp = 0.f;
  float mCosAlp = 0.f;
  float mX = 0.f;    // tracking X
  float mY = 0.f;    // tracking Y
  float mZ = 0.f;    // Z
  float mYRes = 0.f; // tracking Y residual (track - point)
  float mZRes = 0.f; // Z residual
  float mErrYY = 0.f;
  float mErrZZ = 0.f;
  float mErrYZ = 0.f;
  float mSnp = 0.f;

  ClassDefNV(AlgPntDbg, 1);
};

} // namespace align
} // namespace o2
#endif
