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

#ifndef DETECTOR_CALIB_TIMESLOT_H_
#define DETECTOR_CALIB_TIMESLOT_H_

#include <memory>
#include <Rtypes.h>
#include "Framework/Logger.h"
#include "CommonDataFormat/TFIDInfo.h"
#include "CommonConstants/LHCConstants.h"
#include "DetectorsBase/GRPGeomHelper.h"

/// @brief Wrapper for the container of calibration data for single time slot

namespace o2
{
namespace calibration
{

using TFType = uint32_t;
inline constexpr TFType INFINITE_TF = std::numeric_limits<TFType>::max();

template <typename Container>
class TimeSlot
{
 public:
  TimeSlot() = default;
  TimeSlot(TFType tfS, TFType tfE) : mTFStart(tfS), mTFEnd(tfE)
  {
    mTFStartMS = getStartTimeMS();
  }
  TimeSlot(const TimeSlot& src) : mTFStart(src.mTFStart), mTFEnd(src.mTFEnd), mContainer(std::make_unique<Container>(*src.getContainer())) {}
  TimeSlot& operator=(const TimeSlot& src)
  {
    if (&src != this) {
      mTFStart = src.mTFStart;
      mTFEnd = src.mTFEnd;
      mTFStartMS = src.mTFStartMS;
      mContainer = std::make_unique<Container>(*src.getContainer());
    }
    return *this;
  }

  ~TimeSlot() = default;

  TFType getTFStart() const { return mTFStart; }
  TFType getTFEnd() const { return mTFEnd; }

  long getStaticStartTimeMS() const { return mTFStartMS; }
  long getStartTimeMS() const { return o2::base::GRPGeomHelper::instance().getOrbitResetTimeMS() + (mRunStartOrbit + long(o2::base::GRPGeomHelper::getNHBFPerTF()) * mTFStart) * o2::constants::lhc::LHCOrbitMUS / 1000; }
  long getEndTimeMS() const { return o2::base::GRPGeomHelper::instance().getOrbitResetTimeMS() + (mRunStartOrbit + long(o2::base::GRPGeomHelper::getNHBFPerTF()) * (mTFEnd + 1)) * o2::constants::lhc::LHCOrbitMUS / 1000; }

  const Container* getContainer() const { return mContainer.get(); }
  Container* getContainer() { return mContainer.get(); }
  void setContainer(std::unique_ptr<Container> ptr) { mContainer = std::move(ptr); }

  void setTFStart(TFType v) { mTFStart = v; }
  void setTFEnd(TFType v) { mTFEnd = v; }
  void setRunStartOrbit(long t) { mRunStartOrbit = t; }
  auto getRunStartOrbit() const { return mRunStartOrbit; }

  // compare the TF with this slot boundaties
  int relateToTF(TFType tf) { return tf < mTFStart ? -1 : (tf > mTFEnd ? 1 : 0); }

  // merge data of previous slot to this one and extend the mTFStart to cover prev
  void mergeToPrevious(TimeSlot& prev)
  {
    mContainer->merge(prev.mContainer.get());
    mTFStart = prev.mTFStart;
  }

  void print() const
  {
    LOGF(info, "Calibration slot %5d <=TF<=  %5d", mTFStart, mTFEnd);
    mContainer->print();
  }

 private:
  TFType mTFStart = 0;
  TFType mTFEnd = 0;
  size_t mEntries = 0;
  long mRunStartOrbit = 0;
  std::unique_ptr<Container> mContainer; // user object to accumulate the calibration data for this slot
  long mTFStartMS = 0;                   // start time of the slot in ms that avoids to calculate it on the fly; needed when a slot covers more runs, otherwise the OrbitReset that is read is the one of the latest run, and the validity will be wrong

  ClassDefNV(TimeSlot, 2);
};

} // namespace calibration
} // namespace o2

#endif
