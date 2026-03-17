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

#include "IOTOFDigitizerSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Framework/Task.h"
#include "Steer/HitProcessingManager.h"
#include "DataFormatsITSMFT/Digit.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "DetectorsBase/BaseDPLDigitizer.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsCommonDataFormats/SimTraits.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "IOTOFSimulation/Digitizer.h"
#include "Headers/DataHeader.h"

#include <TChain.h>
#include <TStopwatch.h>

#include <algorithm>
#include <memory>
#include <string>

namespace o2::iotof
{

class IOTOFDPLDigitizerTask : o2::base::BaseDPLDigitizer
{
 public:
  using BaseDPLDigitizer::init;

  IOTOFDPLDigitizerTask(bool mctruth = true) : BaseDPLDigitizer(o2::base::InitServices::FIELD | o2::base::InitServices::GEOM),
                                               mWithMCTruth(mctruth) {}

  void initDigitizerTask(framework::InitContext& ic) override
  {
  }

  void run(framework::ProcessingContext& pc)
  {
  }

 private:
  bool mWithMCTruth{true};
};

std::vector<o2::framework::OutputSpec> makeOutChannels(o2::header::DataOrigin detOrig, bool mctruth)
{
  std::vector<o2::framework::OutputSpec> outputs;
  for (uint32_t iLayer = 0; iLayer < 3; ++iLayer) {
    outputs.emplace_back(detOrig, "DIGITS", iLayer, o2::framework::Lifetime::Timeframe);
    outputs.emplace_back(detOrig, "DIGITSROF", iLayer, o2::framework::Lifetime::Timeframe);
    if (mctruth) {
      outputs.emplace_back(detOrig, "DIGITSMC2ROF", iLayer, o2::framework::Lifetime::Timeframe);
      outputs.emplace_back(detOrig, "DIGITSMCTR", iLayer, o2::framework::Lifetime::Timeframe);
    }
  }
  outputs.emplace_back(detOrig, "ROMode", 0, o2::framework::Lifetime::Timeframe);
  return outputs;
}

o2::framework::DataProcessorSpec getIOTOFDigitizerSpec(int channel, bool mctruth)
{
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<o2::header::DataHeader::SubSpecificationType>(channel), o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("IOTOF_aptsresp", "TF3", "APTSRESP", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("IT3/Calib/APTSResponse"));

  const std::string detStr = o2::detectors::DetID::getName(o2::detectors::DetID::TF3);
  return o2::framework::DataProcessorSpec{detStr + "Digitizer",
                                          inputs,
                                          makeOutChannels(o2::header::gDataOriginTF3, mctruth),
                                          o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<IOTOFDPLDigitizerTask>(mctruth)},
                                          o2::framework::Options{
                                            {"disable-qed", o2::framework::VariantType::Bool, false, {"disable QED handling"}},
                                            {"local-response-file", o2::framework::VariantType::String, "", {"use response file saved locally at this path/filename"}}}};
}

} // namespace o2::iotof
