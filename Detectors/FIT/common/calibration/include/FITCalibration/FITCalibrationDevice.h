// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FITCALIBRATIONDEVICE_H
#define O2_FITCALIBRATIONDEVICE_H

#include <utility>

#include "FITCalibration/FITCalibrator.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"

namespace o2::fit
{

template <typename InputCalibrationInfoType, typename TimeSlotStorageType, typename CalibrationObjectType>
class FITCalibrationDevice : public o2::framework::Task
{

  using CalibratorType = FITCalibrator<InputCalibrationInfoType, TimeSlotStorageType, CalibrationObjectType>;

 public:
  explicit FITCalibrationDevice(std::string inputDataLabel, std::string calibrationObjectPath)
    : mInputDataLabel(std::move(inputDataLabel)), mCalibrationObjectPath(std::move(calibrationObjectPath)) {}

  void init(o2::framework::InitContext& context) final;
  void run(o2::framework::ProcessingContext& context) final;
  void endOfStream(o2::framework::EndOfStreamContext& context) final;

 private:
  void _sendOutputs(o2::framework::DataAllocator& outputs);
  void _sendCalibrationObjectIfSlotFinalized(o2::framework::DataAllocator& outputs);

 private:
  std::string mInputDataLabel;
  std::string mCalibrationObjectPath;
  std::unique_ptr<CalibratorType> mCalibrator;
};

#define FIT_CALIBRATION_DEVICE_TEMPLATES \
  template <typename InputCalibrationInfoType, typename TimeSlotStorageType, typename CalibrationObjectType>

#define FIT_CALIBRATION_DEVICE_TYPE \
  FITCalibrationDevice<InputCalibrationInfoType, TimeSlotStorageType, CalibrationObjectType>

FIT_CALIBRATION_DEVICE_TEMPLATES
void FIT_CALIBRATION_DEVICE_TYPE::init(o2::framework::InitContext& context)
{
  mCalibrator = std::make_unique<CalibratorType>(mCalibrationObjectPath);

  //local db instance for testing
  //caching is enabled by default
  o2::ccdb::BasicCCDBManager::instance().setURL("http://localhost:8080");
}

FIT_CALIBRATION_DEVICE_TEMPLATES
void FIT_CALIBRATION_DEVICE_TYPE::run(o2::framework::ProcessingContext& context)
{

  auto TFCounter = o2::header::get<o2::framework::DataProcessingHeader*>(context.inputs().get(mInputDataLabel).header)->startTime;
  auto data = context.inputs().get<gsl::span<InputCalibrationInfoType>>(mInputDataLabel);

  //for testing purposes
  mCalibrator->setCalibrationObject(o2::ccdb::BasicCCDBManager::instance().get<CalibrationObjectType>(mCalibrationObjectPath));

  mCalibrator->process(TFCounter, data);
  _sendCalibrationObjectIfSlotFinalized(context.outputs());
}

FIT_CALIBRATION_DEVICE_TEMPLATES
void FIT_CALIBRATION_DEVICE_TYPE::endOfStream(o2::framework::EndOfStreamContext& context)
{
  //should be finalized even if not enough data?
  mCalibrator->finalizeOldestSlot();
  _sendOutputs(context.outputs());
}

FIT_CALIBRATION_DEVICE_TEMPLATES
void FIT_CALIBRATION_DEVICE_TYPE::_sendCalibrationObjectIfSlotFinalized(o2::framework::DataAllocator& outputs)
{
  if (mCalibrator->isCalibrationObjectReadyToSend()) {
    _sendOutputs(outputs);
  }
}

FIT_CALIBRATION_DEVICE_TEMPLATES
void FIT_CALIBRATION_DEVICE_TYPE::_sendOutputs(o2::framework::DataAllocator& outputs)
{

  using clbUtils = o2::calibration::Utils;

  const auto& payloadVec = mCalibrator->getCalibrationObjectVector();
  auto& infoVec = mCalibrator->getCalibrationInfoVector();
  assert(payloadVec.size() == infoVec.size());

  for (uint32_t i = 0; i < payloadVec.size(); ++i) {
    auto& w = infoVec[i];
    auto image = o2::ccdb::CcdbApi::createObjectImage(&payloadVec[i], &w);
    outputs.snapshot(o2::framework::Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBPayload, i}, *image);
    outputs.snapshot(o2::framework::Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBInfo, i}, w);
  }

  if (!payloadVec.empty()) {
    mCalibrator->initOutput();
  }
}

#undef FIT_CALIBRATION_DEVICE_TEMPLATES
#undef FIT_CALIBRATION_DEVICE_TYPE

} // namespace o2::fit

#endif //O2_FITCALIBRATIONDEVICE_H
