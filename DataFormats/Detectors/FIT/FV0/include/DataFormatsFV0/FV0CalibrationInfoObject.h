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

#ifndef O2_FV0CALIBRATIONINFOOBJECT_H
#define O2_FV0CALIBRATIONINFOOBJECT_H

#include "Rtypes.h"

namespace o2
{
namespace fv0
{
class FV0CalibrationInfoObject
{
 public:
  FV0CalibrationInfoObject(uint8_t channel, int16_t time, int32_t charge, uint64_t timestamp) : mChannelIndex(channel), mTime(time), mCharge(charge), mTimeStamp(timestamp){};
  FV0CalibrationInfoObject() = default;
  ~FV0CalibrationInfoObject() = default;

  void setChannelIndex(uint8_t channel) { mChannelIndex = channel; }
  [[nodiscard]] uint8_t getChannelIndex() const { return mChannelIndex; }

  void setTime(int16_t time) { mTime = time; }
  [[nodiscard]] int16_t getTime() const { return mTime; }
  void setCharge(int16_t charge) { mCharge = charge; }
  [[nodiscard]] int16_t getCharge() const { return mCharge; }
  void setTimeStamp(int64_t timestamp) { mTimeStamp = timestamp; }
  [[nodiscard]] int64_t getTimeStamp() const { return mTimeStamp; }

 private:
  uint8_t mChannelIndex;
  int16_t mTime;
  int16_t mCharge;
  uint64_t mTimeStamp;

  ClassDefNV(FV0CalibrationInfoObject, 1);
};
} // namespace fv0
} // namespace o2

#endif // O2_FV0CALIBRATIONINFOOBJECT_H
