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

#ifndef ALICEO2_TRD_HIT_H_
#define ALICEO2_TRD_HIT_H_

#include <vector>
#include "SimulationDataFormat/BaseHits.h"
#include "CommonUtils/ShmAllocator.h"

namespace o2::trd
{

class Hit : public o2::BasicXYZQHit<float>
{
 public:
  using BasicXYZQHit<float>::BasicXYZQHit;
  Hit(float x, float y, float z, float lCol, float lRow, float lTime, float tof, int charge, int trackId, int detId, bool drift)
    : BasicXYZQHit(x, y, z, tof, charge, trackId, detId), mInDrift(drift), locC(lCol), locR(lRow), locT(lTime){};
  bool isFromDriftRegion() const { return mInDrift; }
  void setLocalC(float lCol) { locC = lCol; }
  void setLocalR(float lRow) { locR = lRow; }
  void setLocalT(float lTime) { locT = lTime; }
  float getLocalC() const { return locC; }
  float getLocalR() const { return locR; }
  float getLocalT() const { return locT; }

 private:
  bool mInDrift{false};
  float locC{-99}; // col direction in amplification or drift volume
  float locR{-99}; // row direction in amplification or drift volume
  float locT{-99}; // time direction in amplification or drift volume

  ClassDefNV(Hit, 1);
};
} // namespace o2::trd

#ifdef USESHM
namespace std
{
template <>
class allocator<o2::trd::Hit> : public o2::utils::ShmAllocator<o2::trd::Hit>
{
};
} // namespace std
#endif

#endif
