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

#ifndef O2_FT0CALIBRATIONINFOOBJECT_H
#define O2_FT0CALIBRATIONINFOOBJECT_H

#include "Rtypes.h"

namespace o2
{
namespace ft0
{
class FT0CalibrationInfoObject
{
 public:
  FT0CalibrationInfoObject(uint8_t channel, int16_t time, int32_t amp, uint64_t timestamp) : mChannelIndex(channel), mTime(time), mAmp(amp), mTimeStamp(timestamp){};
  FT0CalibrationInfoObject() = default;
  ~FT0CalibrationInfoObject() = default;

  void setChannelIndex(uint8_t channel) { mChannelIndex = channel; }
  [[nodiscard]] uint8_t getChannelIndex() const { return mChannelIndex; }

  void setTime(int16_t time) { mTime = time; }
  [[nodiscard]] int16_t getTime() const { return mTime; }
  void setAmp(int16_t amp) { mAmp = amp; }
  [[nodiscard]] int16_t getAmp() const { return mAmp; }
  void setTimeStamp(int64_t timestamp) { mTimeStamp = timestamp; }
  [[nodiscard]] int64_t getTimeStamp() const { return mTimeStamp; }

 private:
  uint8_t mChannelIndex;
  int16_t mTime;
  int16_t mAmp;
  uint64_t mTimeStamp;

  ClassDefNV(FT0CalibrationInfoObject, 2);
};
} // namespace ft0
} // namespace o2

#endif // O2_FT0CALIBRATIONINFOOBJECT_H
