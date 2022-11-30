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

#ifndef O2_FITCALIBRATOR_H
#define O2_FITCALIBRATOR_H

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DetectorsCalibration/Utils.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CCDB/BasicCCDBManager.h"
#include "DetectorsRaw/HBFUtils.h"
#include "Rtypes.h"
#include <type_traits>
#include <vector>

namespace o2::fit
{

template <typename TimeSlotStorageType, typename CalibrationObjectType>
class FITCalibrator final : public o2::calibration::TimeSlotCalibration<TimeSlotStorageType>
{

  // probably will be set via run parameter
  static constexpr unsigned int DEFAULT_MIN_ENTRIES = 1000;
  using CalibObjWithInfoType = std::pair<o2::ccdb::CcdbObjectInfo, std::unique_ptr<std::vector<char>>>;
  using TFType = o2::calibration::TFType;
  using Slot = o2::calibration::TimeSlot<TimeSlotStorageType>;

 public:
  explicit FITCalibrator(unsigned int minimumEntries = DEFAULT_MIN_ENTRIES) : mMinEntries(minimumEntries) // should be removed in future
  {
    LOG(debug) << "FITCalibrator ";
  }
  ~FITCalibrator() final = default;

  [[nodiscard]] bool isCalibrationObjectReadyToSend() const { return !mStoredCalibrationObjects.empty(); }
  [[nodiscard]] const std::vector<std::pair<o2::ccdb::CcdbObjectInfo, std::unique_ptr<std::vector<char>>>>& getStoredCalibrationObjects() const { return mStoredCalibrationObjects; }

  bool hasEnoughData(const Slot& slot) const final
  {
    LOG(info) << "FIT_CALIBRATOR_TYPE::hasEnoughData";
    return slot.getContainer()->hasEnoughEntries();
  }

  void initOutput() final
  {
    LOG(info) << "FIT_CALIBRATOR_TYPE::initOutput";
    mStoredCalibrationObjects.clear();
  }

  void finalizeSlot(Slot& slot) final
  {
    static std::map<std::string, std::string> md;
    auto* container = slot.getContainer();
    static const double TFlength = 1E-3 * o2::raw::HBFUtils::Instance().getNOrbitsPerTF() * o2::constants::lhc::LHCOrbitMUS; // in ms
    auto starting = slot.getStartTimeMS();
    auto stopping = slot.getEndTimeMS();
    LOGP(info, "!!!! {}({})<=TF<={}({}), starting: {} stopping {}", slot.getTFStart(), slot.getStartTimeMS(), slot.getTFEnd(), slot.getEndTimeMS(), starting, stopping);
    auto calibrationObject = container->generateCalibrationObject();
    std::vector<CalibObjWithInfoType> preparedCalibObjects;
    preparedCalibObjects.emplace_back(doSerializationAndPrepareObjectInfo(calibrationObject, starting, stopping));
    mStoredCalibrationObjects.insert(mStoredCalibrationObjects.end(),
                                     std::make_move_iterator(preparedCalibObjects.begin()),
                                     std::make_move_iterator(preparedCalibObjects.end()));
  }

  Slot& emplaceNewSlot(
    bool front, TFType tstart, TFType tend) final
  {
    LOG(info) << "FIT_CALIBRATOR_TYPE::emplaceNewSlot "
              << " start " << tstart << " end " << tend;
    auto& cont = o2::calibration::TimeSlotCalibration<TimeSlotStorageType>::getSlots();
    auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
    slot.setContainer(std::make_unique<TimeSlotStorageType>(mMinEntries));
    return slot;
  }

  CalibObjWithInfoType doSerializationAndPrepareObjectInfo(const CalibrationObjectType& calibrationObject, long starting, long stopping)
  {
    std::map<std::string, std::string> metaData;
    CalibObjWithInfoType result;
    auto clName = o2::utils::MemFileHelper::getClassName(calibrationObject);
    auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
    LOG(info) << " clName " << clName << " flName " << flName;
    result.first = o2::ccdb::CcdbObjectInfo(CalibrationObjectType::getObjectPath(), clName, flName, metaData, starting, stopping);
    result.second = o2::ccdb::CcdbApi::createObjectImage(&calibrationObject, &result.first);
    LOG(info) << "doSerializationAndPrepareObjectInfo"
              << " start " << starting << " end " << stopping;
    return result;
  }

 private:
  std::vector<CalibObjWithInfoType> mStoredCalibrationObjects{};
  const unsigned int mMinEntries;
};

} // namespace o2::fit

#endif // O2_FITCALIBRATOR_H
