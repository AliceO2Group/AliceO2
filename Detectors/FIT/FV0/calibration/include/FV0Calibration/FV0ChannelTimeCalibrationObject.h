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

#ifndef O2_FV0CHANNELTIMECALIBRATIONOBJECT_H
#define O2_FV0CHANNELTIMECALIBRATIONOBJECT_H

#include <array>
#include "Rtypes.h"
#include "DataFormatsFV0/RawEventData.h"
#include "FV0Base/Constants.h"

namespace o2::fv0
{

struct FV0ChannelTimeCalibrationObject {

  std::array<int16_t, Constants::nFv0Channels> mTimeOffsets{};

  ClassDefNV(FV0ChannelTimeCalibrationObject, 1);
};

class FV0ChannelTimeTimeSlotContainer;

class FV0TimeChannelOffsetCalibrationObjectAlgorithm
{
 public:
  [[nodiscard]] static FV0ChannelTimeCalibrationObject generateCalibrationObject(const FV0ChannelTimeTimeSlotContainer& container);
};

} // namespace o2::fv0

#endif //O2_FV0CHANNELTIMECALIBRATIONOBJECT_H
