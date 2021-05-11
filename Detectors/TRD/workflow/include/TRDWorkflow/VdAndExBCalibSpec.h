// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRD_VDANDEXBCALIBSPEC_H
#define O2_TRD_VDANDEXBCALIBSPEC_H

/// \file   VdAndExBCalibSpec.h
/// \brief DPL device for steering the TRD vD and ExB time slot based calibration
/// \author Ole Schmidt

#include "TRDCalibration/CalibratorVdExB.h"
#include "DetectorsCalibration/Utils.h"
#include "DataFormatsTRD/AngularResidHistos.h"
#include "CommonUtils/MemFileHelper.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"

using namespace o2::framework;

namespace o2
{
namespace calibration
{

class VdAndExBCalibDevice : public o2::framework::Task
{
 public:
  void init(o2::framework::InitContext& ic) final
  {
    int minEnt = std::max(50'000, ic.options().get<int>("min-entries"));
    int slotL = ic.options().get<int>("tf-per-slot");
    int delay = ic.options().get<int>("max-delay");
    mCalibrator = std::make_unique<o2::trd::CalibratorVdExB>(minEnt);
    mCalibrator->setSlotLength(slotL);
    mCalibrator->setMaxSlotsDelay(delay);
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("input").header)->startTime;
    auto data = pc.inputs().get<gsl::span<o2::trd::AngularResidHistos>>("input");
    LOG(INFO) << "Processing TF " << tfcounter << " with " << data.size() << " AngularResidHistos objects";
    mCalibrator->process(tfcounter, data);
    sendOutput(pc.outputs());
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOG(INFO) << "Finalizing calibration";
    constexpr uint64_t INFINITE_TF = 0xffffffffffffffff;
    mCalibrator->checkSlotsToFinalize(INFINITE_TF);
    sendOutput(ec.outputs());
  }

 private:
  std::unique_ptr<o2::trd::CalibratorVdExB> mCalibrator;

  //________________________________________________________________
  void sendOutput(DataAllocator& output)
  {
    // See LHCClockCalibratorSpec.h
    // Before this can be implemented the output CCDB objects need to be defined
    // and added to CalibratorVdExB
  }
};

} // namespace calibration

namespace framework
{

DataProcessorSpec getTRDVdAndExBCalibSpec()
{
  using device = o2::calibration::VdAndExBCalibDevice;
  using clbUtils = o2::calibration::Utils;

  std::vector<OutputSpec> outputs;
  //outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBPayload});
  //outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBInfo});
  return DataProcessorSpec{
    "calib-vdexb-calibration",
    Inputs{{"input", "TRD", "ANGRESHISTS"}},
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{
      {"tf-per-slot", VariantType::Int, 5, {"number of TFs per calibration time slot"}},
      {"max-delay", VariantType::Int, 90'000, {"number of slots in past to consider"}}, // 15 minutes delay, 10ms TF
      {"min-entries", VariantType::Int, 500, {"minimum number of entries to fit single time slot"}}}};
}

} // namespace framework
} // namespace o2

#endif // O2_TRD_VDANDEXBCALIBSPEC_H
