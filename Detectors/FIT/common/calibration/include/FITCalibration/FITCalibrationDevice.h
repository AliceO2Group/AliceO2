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

#define FIT_CALIBRATION_DEVICE_TEMPLATES \
  template <typename InputCalibrationInfoType, typename TimeSlotStorageType, typename CalibrationObjectType>

#define FIT_CALIBRATION_DEVICE_TYPE \
  FITCalibrationDevice<InputCalibrationInfoType, TimeSlotStorageType, CalibrationObjectType>

FIT_CALIBRATION_DEVICE_TEMPLATES
class FITCalibrationDevice : public o2::framework::Task
{

  static constexpr const char* DEFAULT_INPUT_DATA_LABEL = "calib";
  using CalibratorType = FITCalibrator<InputCalibrationInfoType, TimeSlotStorageType, CalibrationObjectType>;

 public:
  explicit FITCalibrationDevice(std::string inputDataLabel = DEFAULT_INPUT_DATA_LABEL)
    : mInputDataLabel(std::move(inputDataLabel)) {}

  void init(o2::framework::InitContext& context) final;
  void run(o2::framework::ProcessingContext& context) final;
  void endOfStream(o2::framework::EndOfStreamContext& context) final;

 private:
  void _sendOutputs(o2::framework::DataAllocator& outputs);
  void _sendCalibrationObjectIfSlotFinalized(o2::framework::DataAllocator& outputs);

 private:
  const std::string mInputDataLabel;
  std::unique_ptr<CalibratorType> mCalibrator;
};

FIT_CALIBRATION_DEVICE_TEMPLATES
void FIT_CALIBRATION_DEVICE_TYPE::init(o2::framework::InitContext& context)
{
  mCalibrator = std::make_unique<CalibratorType>();
  FITCalibrationApi::init();
}

FIT_CALIBRATION_DEVICE_TEMPLATES
void FIT_CALIBRATION_DEVICE_TYPE::run(o2::framework::ProcessingContext& context)
{

  auto TFCounter = o2::header::get<o2::framework::DataProcessingHeader*>(context.inputs().get(mInputDataLabel).header)->startTime;
  auto data = context.inputs().get<gsl::span<InputCalibrationInfoType>>(mInputDataLabel);

  //something like that probably in the future
  //  FITCalibrationApi::setProcessingTimestamp( getTimestampForTF(TFCounter)) );

  mCalibrator->process(TFCounter, data);
  _sendCalibrationObjectIfSlotFinalized(context.outputs());
}

FIT_CALIBRATION_DEVICE_TEMPLATES
void FIT_CALIBRATION_DEVICE_TYPE::endOfStream(o2::framework::EndOfStreamContext& context)
{
  //nope, we have to check if we can finalize slot anyway - scenario with one batch
  static constexpr uint64_t INFINITE_TF = 0xffffffffffffffff;
  mCalibrator->checkSlotsToFinalize(INFINITE_TF);
  _sendCalibrationObjectIfSlotFinalized(context.outputs());
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
  const auto& objectsToSend = mCalibrator->getStoredCalibrationObjects();

  uint32_t iSendChannel = 0;
  for (const auto& [ccdbInfo, calibObject] : objectsToSend) {
    outputs.snapshot(o2::framework::Output{clbUtils::gDataOriginCDBPayload, "FIT_CALIB", iSendChannel}, *calibObject);
    outputs.snapshot(o2::framework::Output{clbUtils::gDataOriginCDBWrapper, "FIT_CALIB", iSendChannel}, ccdbInfo);
    ++iSendChannel;
  }
  mCalibrator->initOutput();
}

#undef FIT_CALIBRATION_DEVICE_TEMPLATES
#undef FIT_CALIBRATION_DEVICE_TYPE

} // namespace o2::fit

#endif //O2_FITCALIBRATIONDEVICE_H
