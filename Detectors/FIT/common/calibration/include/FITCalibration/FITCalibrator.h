// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FITCALIBRATOR_H
#define O2_FITCALIBRATOR_H

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DetectorsCalibration/Utils.h"
#include "CommonUtils/MemFileHelper.h"
#include "Rtypes.h"
#include "FITCalibration/FITCalibrationObjectProducer.h"
#include "FITCalibration/FITCalibrationApi.h"

namespace o2::fit
{

#define FIT_CALIBRATOR_TEMPLATES \
  template <typename InputCalibrationInfoType, typename TimeSlotStorageType, typename CalibrationObjectType>

#define FIT_CALIBRATOR_TYPE \
  FITCalibrator<InputCalibrationInfoType, TimeSlotStorageType, CalibrationObjectType>


FIT_CALIBRATOR_TEMPLATES
class FITCalibrator final : public o2::calibration::TimeSlotCalibration<InputCalibrationInfoType, TimeSlotStorageType>
{

  //probably will be set via run parameter
  static constexpr unsigned int DEFAULT_MIN_ENTRIES = 10000;

  //temp param for testing
  static constexpr bool DEFAULT_TEST_MODE = true;

  using TFType = uint64_t;
  using Slot = o2::calibration::TimeSlot<TimeSlotStorageType>;

 public:
  explicit FITCalibrator(unsigned int minimumEntries = DEFAULT_MIN_ENTRIES, bool testMode = DEFAULT_TEST_MODE);

  ~FITCalibrator() final = default;

  [[nodiscard]] bool hasEnoughData(const Slot& slot) const final;
  void initOutput() final;
  void finalizeSlot(Slot& slot) final;
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;

  [[nodiscard]] bool isCalibrationObjectReadyToSend() const { return !mCalibrationObjectVector.empty(); }
  [[nodiscard]] const std::vector<CalibrationObjectType>& getCalibrationObjectVector() const { return mCalibrationObjectVector; }
  [[nodiscard]] const std::vector<o2::ccdb::CcdbObjectInfo>& getCalibrationInfoVector() const { return mInfoVector; }
  [[nodiscard]] std::vector<o2::ccdb::CcdbObjectInfo>& getCalibrationInfoVector() { return mInfoVector; }

 private:
  [[nodiscard]] bool _isTestModeEnabled() const { return mTestMode; }

 private:
  std::vector<o2::ccdb::CcdbObjectInfo> mInfoVector;
  std::vector<CalibrationObjectType> mCalibrationObjectVector;
  const unsigned int mMinEntries;
  const bool mTestMode;

};



FIT_CALIBRATOR_TEMPLATES
FIT_CALIBRATOR_TYPE::FITCalibrator(unsigned int minimumEntries, bool testMode)
  : mMinEntries(minimumEntries), mTestMode(testMode)
{
}

FIT_CALIBRATOR_TEMPLATES
bool FIT_CALIBRATOR_TYPE::hasEnoughData(const Slot& slot) const
{
  if(_isTestModeEnabled()){
    static unsigned int testCounter = 0;
    ++testCounter;
    if(!(testCounter % 1000)){
      return true;
    }
  }

  return slot.getContainer()->hasEnoughEntries();
}

FIT_CALIBRATOR_TEMPLATES
void FIT_CALIBRATOR_TYPE::initOutput()
{
  mInfoVector.clear();
  mCalibrationObjectVector.clear();
}

FIT_CALIBRATOR_TEMPLATES
void FIT_CALIBRATOR_TYPE::finalizeSlot(Slot& slot)
{
  static std::map<std::string, std::string> md;

  const auto& container = slot.getContainer();
  auto calibrationObject = FITCalibrationObjectProducer::generateCalibrationObject<CalibrationObjectType>(*container);
  auto clName = o2::utils::MemFileHelper::getClassName(calibrationObject);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  // end of validity = -1 means currentTimestamp + 1 year
  mInfoVector.emplace_back(FITCalibrationApi::getObjectPath<CalibrationObjectType>(), clName, flName, md, ccdb::getCurrentTimestamp(), -1);
  mCalibrationObjectVector.emplace_back(calibrationObject);
}

FIT_CALIBRATOR_TEMPLATES
typename FIT_CALIBRATOR_TYPE::Slot& FIT_CALIBRATOR_TYPE::emplaceNewSlot(
  bool front, TFType tstart, TFType tend)
{
  auto& cont = o2::calibration::TimeSlotCalibration<InputCalibrationInfoType, TimeSlotStorageType>::getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<TimeSlotStorageType>(mMinEntries));
  return slot;
}


#undef FIT_CALIBRATOR_TEMPLATES
#undef FIT_CALIBRATOR_TYPE

} // namespace o2::fit

#endif //O2_FITCALIBRATOR_H
