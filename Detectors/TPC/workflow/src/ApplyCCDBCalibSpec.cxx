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

#include <vector>

#include "Framework/CCDBParamSpec.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Logger.h"

#include "TPCBase/ParameterGas.h"
#include "DataFormatsTPC/LtrCalibData.h"
#include "TPCWorkflow/ApplyCCDBCalibSpec.h"
#include <boost/property_tree/ptree.hpp>

using namespace o2::framework;

namespace o2::tpc
{

class ApplyCCDBCalibDevice : public o2::framework::Task
{
 public:
  void init(InitContext& ic) final
  {
    //const int minEnt = ic.options().get<int>("min-tfs");
    mReferenceDriftV = o2::conf::ConfigurableParam::getValueAs<float>("TPCGasParam.DriftV");
    LOGP(info, "Setting reference drift velocity to {}", mReferenceDriftV);
  }

  void run(ProcessingContext& pc) final
  {
    applyLaserVDrift(pc.inputs());
  }

 private:
  float mReferenceDriftV{};                          ///< reference drift velocity to apply correction to
  float mDriftV{};                                   ///< present drift velocity
  std::unique_ptr<const LtrCalibData> mLtrCalibData; ///< presently used laser track calibration

  void applyLaserVDrift(InputRecord& inputs);
  void updateVDrift();
};

void ApplyCCDBCalibDevice::applyLaserVDrift(InputRecord& inputs)
{
  auto ltrCalibPtr = inputs.get<LtrCalibData*>("calibLaserTracks");
  if (!mLtrCalibData) {
    mLtrCalibData.reset(ltrCalibPtr.release());
    LOGP(info, "Laser track calibration initialized: {} - {}: A - {} C - {}", mLtrCalibData->firstTime, mLtrCalibData->lastTime, mLtrCalibData->dvCorrectionA, mLtrCalibData->dvCorrectionC);
    updateVDrift();
  } else {
    if (ltrCalibPtr->firstTime != mLtrCalibData->firstTime) {
      mLtrCalibData.reset(ltrCalibPtr.release());
      LOGP(info, "Laser track calibration updated: {} - {}: A - {} C - {}, was {} - {}: A - {} C - {}", mLtrCalibData->firstTime, mLtrCalibData->lastTime, mLtrCalibData->dvCorrectionA, mLtrCalibData->dvCorrectionC, ltrCalibPtr->firstTime, ltrCalibPtr->lastTime, ltrCalibPtr->dvCorrectionA, ltrCalibPtr->dvCorrectionC);
      updateVDrift();
    }
  }
}

void ApplyCCDBCalibDevice::updateVDrift()
{
  const float oldDriftV = mDriftV;
  mDriftV = mReferenceDriftV * mLtrCalibData->getDriftVCorrection();
  LOGP(info, "updating drift velocity correction to {}, was {}, reference is {} (cm / us)", mDriftV, oldDriftV, mReferenceDriftV);
  o2::conf::ConfigurableParam::setValue<float>("TPCGasParam", "DriftV", mDriftV);
}

//______________________________________________________________________________
DataProcessorSpec getApplyCCDBCalibSpec()
{
  using device = ApplyCCDBCalibDevice;

  std::vector<InputSpec> inputs{
    InputSpec{"calibLaserTracks", "TPC", "CalibLaserTracks", 0, Lifetime::Condition, ccdbParamSpec("TPC/Calib/LaserTracks")},
    InputSpec{"sometimer", "TST", "BAR", 0, Lifetime::Timer, {startTimeParamSpec(1638548475371)}},
  };

  return DataProcessorSpec{
    "tpc-apply-ccdb-calib",
    inputs,
    Outputs{},
    AlgorithmSpec{adaptFromTask<device>()},
    Options{
      //{"min-tfs", VariantType::Int, 100, {"minimum number of TFs with enough laser tracks to finalize a slot"}},
    }};
}

} // namespace o2::tpc
