// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CosmicInfo.h
/// \brief Info from cosmic

#ifndef ALICEO2_TOF_COSMICINFO_H
#define ALICEO2_TOF_COSMICINFO_H

#include <vector>
#include "Rtypes.h"

namespace o2
{
namespace tof
{
/// \class CalibInfoCluster
/// \brief CalibInfoCluster for TOF
///
class CosmicInfo
{
  int mChan1;
  int mChan2;
  float mDtime;
  float mTot1;
  float mTot2;
  float mL;
  float mT1;
  float mT2;

 public:
  int getCH1() const { return mChan1; }
  int getCH2() const { return mChan2; }
  float getDeltaTime() const { return mDtime; }
  float getTOT1() const { return mTot1; }
  float getTOT2() const { return mTot2; }
  float getL() const { return mL; }
  float getT1() const { return mT1; }
  float getT2() const { return mT2; }

  void setCH1(int ch) { mChan1 = ch; }
  void setCH2(int ch) { mChan2 = ch; }
  void setDeltaTime(float val) { mDtime = val; }
  void setTOT1(float val) { mTot1 = val; }
  void setTOT2(float val) { mTot2 = val; }
  void setT1(float val) { mT1 = val; }
  void setT2(float val) { mT2 = val; }
  void setL(float val) { mL = val; }

  CosmicInfo(int ch1 = 0, int ch2 = 0, float dt = 0, float tot1 = 0, float tot2 = 0, float l = 0, float tm1 = 0, float tm2 = 0) : mChan1(ch1), mChan2(ch2), mDtime(dt), mTot1(tot1), mTot2(tot2), mL(l), mT1(tm1), mT2(tm2) {}

  ClassDefNV(CosmicInfo, 2);
};

class CalibInfoTrackCl
{
  int mCh = 0;
  float mX = 0;
  float mY = 0;
  float mZ = 0;
  float mT = 0;
  short mTot = 0;

 public:
  int getCH() const { return mCh; }
  float getT() const { return mT; }
  float getX() const { return mX; }
  float getY() const { return mY; }
  float getZ() const { return mZ; }
  short getTOT() const { return mTot; }

  CalibInfoTrackCl() = default;
  CalibInfoTrackCl(int ch, float x, float y, float z, float t, short tot) : mCh(ch), mX(x), mY(y), mZ(z), mT(t), mTot(tot) {}
  ClassDefNV(CalibInfoTrackCl, 1);
};

} // namespace tof

} // namespace o2

#endif
