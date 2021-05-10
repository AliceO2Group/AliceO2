// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//Dummy calibration object to present usage of FIT calibration module
//Should be deleted if no more needed as an example

#ifndef O2_FT0DUMMYCALIBRATIONOBJECT_H
#define O2_FT0DUMMYCALIBRATIONOBJECT_H

#include <array>
#include "Rtypes.h"
#include "DataFormatsFT0/RawEventData.h"

namespace o2::ft0
{

struct FT0DummyCalibrationObjectTime {

  std::array<int16_t, o2::ft0::Nchannels_FT0> mStorage{};
  ClassDefNV(FT0DummyCalibrationObjectTime, 1);
};

struct FT0DummyCalibrationObjectCharge {
  std::array<int16_t, o2::ft0::Nchannels_FT0> mStorage{};
  ClassDefNV(FT0DummyCalibrationObjectCharge, 1);
};

struct FT0DummyNeededCalibrationObject {
  std::array<int16_t, o2::ft0::Nchannels_FT0> mStorage{};
  ClassDefNV(FT0DummyNeededCalibrationObject, 1);
};

//lets assume this object contains other calibration objects
struct FT0DummyCalibrationObject {

  FT0DummyCalibrationObjectTime mTimeCalibrationObject;
  FT0DummyCalibrationObjectCharge mChargeCalibrationObject;

  ClassDefNV(FT0DummyCalibrationObject, 1);
};

//lets use existing container (which can be also modified if needed)
class FT0ChannelTimeTimeSlotContainer;

class FT0DummyCalibrationObjectAlgorithm
{
 public:
  [[nodiscard]] static FT0DummyCalibrationObject generateCalibrationObject(const FT0ChannelTimeTimeSlotContainer& container, const FT0DummyNeededCalibrationObject& additionalObjectNeededForCalibration)
  {
    //(add here your printing stuff for debug)
    //do nothing
    (void)additionalObjectNeededForCalibration;
    return {};
  }
};

} // namespace o2::ft0

#endif //O2_FT0DUMMYCALIBRATIONOBJECT_H
