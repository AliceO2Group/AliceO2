// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef DETECTOR_CALIB_TIMESLOT_H_
#define DETECTOR_CALIB_TIMESLOT_H_

#include <memory>
#include <Rtypes.h>
#include "Framework/Logger.h"

/// @brief Wrapper for the container of calibration data for single time slot

namespace o2
{
namespace calibration
{

using TFType = uint64_t;

template <typename Container>
class TimeSlot
{
 public:
  TimeSlot() = default;
  TimeSlot(TFType tfS, TFType tfE) : mTFStart(tfS), mTFEnd(tfE) {}
  TimeSlot(const TimeSlot& src) : mTFStart(src.mTFStart), mTFEnd(src.mTFEnd), mContainer(std::make_unique<Container>(*src.getContainer())) {}
  TimeSlot& operator=(const TimeSlot& src)
  {
    if (&src != this) {
      mTFStart = src.mTFStart;
      mTFEnd = src.mTFEnd;
      mContainer = std::make_unique<Container>(*src.getContainer());
    }
    return *this;
  }

  ~TimeSlot() = default;

  TFType getTFStart() const { return mTFStart; }
  TFType getTFEnd() const { return mTFEnd; }
  const Container* getContainer() const { return mContainer.get(); }
  Container* getContainer() { return mContainer.get(); }
  void setContainer(std::unique_ptr<Container> ptr) { mContainer = std::move(ptr); }

  void setTFStart(TFType v) { mTFStart = v; }
  void setTFEnd(TFType v) { mTFEnd = v; }

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
    LOGF(INFO, "Calibration slot %5d <=TF<=  %5d", mTFStart, mTFEnd);
    mContainer->print();
  }

 private:
  TFType mTFStart = 0;
  TFType mTFEnd = 0;
  size_t mEntries = 0;
  std::unique_ptr<Container> mContainer; // user object to accumulate the calibration data for this slot

  ClassDefNV(TimeSlot, 1);
};

} // namespace calibration
} // namespace o2

#endif
