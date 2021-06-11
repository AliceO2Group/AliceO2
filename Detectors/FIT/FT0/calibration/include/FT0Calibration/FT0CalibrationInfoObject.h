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

namespace o2::ft0
{
class FT0CalibrationInfoObject
{
 public:
  FT0CalibrationInfoObject(uint8_t channel, int16_t time) : mChannelIndex(channel), mTime(time){};
  FT0CalibrationInfoObject() = default;
  ~FT0CalibrationInfoObject() = default;

  void setChannelIndex(uint8_t channel) { mChannelIndex = channel; }
  [[nodiscard]] uint8_t getChannelIndex() const { return mChannelIndex; }

  void setTime(int16_t time) { mTime = time; }
  [[nodiscard]] int16_t getTime() const { return mTime; }

 private:
  uint8_t mChannelIndex;
  int16_t mTime;

  ClassDefNV(FT0CalibrationInfoObject, 1);
};
} // namespace o2::ft0

#endif //O2_FT0CALIBRATIONINFOOBJECT_H
