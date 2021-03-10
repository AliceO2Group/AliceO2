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

 public:
  int getCH1() const { return mChan1; }
  int getCH2() const { return mChan2; }
  float getDeltaTime() const { return mDtime; }
  float getTOT1() const { return mTot1; }
  float getTOT2() const { return mTot2; }
  float getL() const { return mL; }

  void setCH1(int ch) { mChan1 = ch; }
  void setCH2(int ch) { mChan2 = ch; }
  void setDeltaTime(float val) { mDtime = val; }
  void setTOT1(float val) { mTot1 = val; }
  void setTOT2(float val) { mTot2 = val; }
  void setL(float val) { mL = val; }

  CosmicInfo(int ch1 = 0, int ch2 = 0, float dt = 0, float tot1 = 0, float tot2 = 0, float l = 0) : mChan1(ch1), mChan2(ch2), mDtime(dt), mTot1(tot1), mTot2(tot2), mL(l) {}

  ClassDefNV(CosmicInfo, 1);
};
} // namespace tof

} // namespace o2

#endif
