// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  FT0CalibrationInfoObject(uint8_t channel, int16_t time, int32_t amp) : mChannelIndex(channel), mTime(time), mAmp(amp){};
  FT0CalibrationInfoObject() = default;
  ~FT0CalibrationInfoObject() = default;

  void setChannelIndex(uint8_t channel) { mChannelIndex = channel; }
  [[nodiscard]] uint8_t getChannelIndex() const { return mChannelIndex; }

  void setTime(int8_t time) { mTime = time; }
  [[nodiscard]] int8_t getTime() const { return mTime; }
  void setAmp(int16_t amp) { mAmp = amp; }
  [[nodiscard]] int16_t getAmp() const { return mAmp; }

 private:
  uint8_t mChannelIndex;
  int16_t mTime;
  int16_t mAmp;

  ClassDefNV(FT0CalibrationInfoObject, 2);
};
} // namespace ft0
} // namespace o2

#endif //O2_FT0CALIBRATIONINFOOBJECT_H
