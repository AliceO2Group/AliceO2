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

#ifndef O2_TRD_NOISECALIBAGGREGATORSPEC_H
#define O2_TRD_NOISECALIBAGGREGATORSPEC_H

/// \file   NoiseCalibAggregatorSpec.h
/// \brief Create TRD noise map for CCDB from mean ADC values per pad

#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "TRDWorkflow/NoiseCalibSpec.h"
#include "TRDCalibration/CalibratorNoise.h"
#include "DataFormatsTRD/NoiseCalibration.h"

using namespace o2::framework;

namespace o2
{
namespace trd
{

class TRDNoiseCalibAggregatorSpec : public o2::framework::Task
{
 public:
  TRDNoiseCalibAggregatorSpec() = default;
  void init(o2::framework::InitContext& ic) final
  {
    mCalibrator = std::make_unique<o2::trd::CalibratorNoise>();
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto data = pc.inputs().get<PadAdcInfo>("padadcs");
    o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mCalibrator->getCurrentTFInfo());
    mCalibrator->process(data);
    sendOutput(pc.outputs());
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOG(info) << "Finalizing calibration";
    mCalibrator->checkSlotsToFinalize(o2::calibration::INFINITE_TF);
    sendOutput(ec.outputs());
  }

 private:
  std::unique_ptr<o2::trd::CalibratorNoise> mCalibrator;
  //
  void sendOutput(DataAllocator& output)
  {
    // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
    // TODO in principle, this routine is generic, can be moved to Utils.h

    using clbUtils = o2::calibration::Utils;
    const auto& payloadVec = mCalibrator->getCcdbObjectVector();
    auto& infoVec = mCalibrator->getCcdbObjectInfoVector(); // use non-const version as we update it
    assert(payloadVec.size() == infoVec.size());

    for (uint32_t i = 0; i < payloadVec.size(); i++) {
      auto& w = infoVec[i];
      auto image = o2::ccdb::CcdbApi::createObjectImage(&payloadVec[i], &w);
      LOG(info) << "Sending object " << w.getPath() << "/" << w.getFileName() << " of size " << image->size()
                << " bytes, valid for " << w.getStartValidityTimestamp() << " : " << w.getEndValidityTimestamp();

      output.snapshot(Output{clbUtils::gDataOriginCDBPayload, "PADSTATUS", i}, *image.get()); // vector<char>
      output.snapshot(Output{clbUtils::gDataOriginCDBWrapper, "PADSTATUS", i}, w);            // root-serialized
    }
    if (payloadVec.size()) {
      mCalibrator->initOutput(); // reset the outputs once they are already sent
    }
  }
};

o2::framework::DataProcessorSpec getTRDNoiseCalibAggregatorSpec()
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "PADSTATUS"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "PADSTATUS"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "trd-noise-calib-aggregator",
    Inputs{{"padadcs", o2::header::gDataOriginTRD, "PADADCS", 0, Lifetime::Timeframe}},
    outputs,
    AlgorithmSpec{adaptFromTask<TRDNoiseCalibAggregatorSpec>()},
    Options{}};
}

} // namespace trd
} // namespace o2

#endif // O2_TRD_NOISECALIBAGGREGATORSPEC_H
