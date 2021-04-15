// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FT0CALIBRATIONOBJECT_H
#define O2_FT0CALIBRATIONOBJECT_H

#include <array>
#include "Rtypes.h"
#include "DataFormatsFT0/RawEventData.h"
#include "CommonUtils/MemFileHelper.h"
#include "TGraph.h"

namespace o2::ft0
{

struct FT0CalibrationObject {

  std::array<int16_t, o2::ft0::Nchannels_FT0> mChannelOffsets{};

  ClassDefNV(FT0CalibrationObject, 1);
};

class FT0ChannelDataTimeSlotContainer;

class FT0CalibrationObjectAlgorithm
{
 public:
  static void calibrate(FT0CalibrationObject& calibrationObject, const FT0ChannelDataTimeSlotContainer& container);
};

class FT0CalibrationObjectConverter
{
 public:
  static std::unique_ptr<TGraph> toTGraph(const FT0CalibrationObject& object);
};

} // namespace o2::ft0

#endif //O2_FT0CALIBRATIONOBJECT_H
