// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FT0CHANNELTIMECALIBRATIONOBJECT_H
#define O2_FT0CHANNELTIMECALIBRATIONOBJECT_H

#include <array>
#include "Rtypes.h"
#include "DataFormatsFT0/RawEventData.h"

namespace o2::ft0
{

struct FT0ChannelTimeCalibrationObject {

  std::array<int16_t, o2::ft0::Nchannels_FT0> mTimeOffsets{};

  ClassDefNV(FT0ChannelTimeCalibrationObject, 1);
};

class FT0ChannelTimeTimeSlotContainer;

class FT0TimeChannelOffsetCalibrationObjectAlgorithm
{
 public:
  [[nodiscard]] static FT0ChannelTimeCalibrationObject generateCalibrationObject(const FT0ChannelTimeTimeSlotContainer& container);
};

} // namespace o2::ft0

#endif //O2_FT0CHANNELTIMECALIBRATIONOBJECT_H
