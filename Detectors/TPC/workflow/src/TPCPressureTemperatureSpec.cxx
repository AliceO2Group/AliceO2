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

/// \file TPCPressureTemperatureSpec.cxx
/// \brief device for providing pressure and temperature values
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Jun 4, 2025

#include "TPCWorkflow/TPCPressureTemperatureSpec.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "TPCCalibration/PressureTemperatureHelper.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{

class PressureTemperatureDevice : public o2::framework::Task
{
 public:
  PressureTemperatureDevice(std::shared_ptr<o2::base::GRPGeomRequest> req) : mCCDBRequest(req){};
  void init(o2::framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    const int intInterval = ic.options().get<int>("fit-interval");
    mPTHelper.setFitIntervalTemp(intInterval * 1000);
    const bool enableDebugTree = ic.options().get<bool>("enable-root-output");
    if (enableDebugTree) {
      mStreamer = std::make_unique<o2::utils::TreeStreamRedirector>("pt.root", "recreate");
    }
  };

  void endOfStream(EndOfStreamContext& eos) final
  {
    if (mStreamer) {
      mStreamer->Close();
    }
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    mPTHelper.extractCCDBInputs(pc);
    const auto orbitResetTimeMS = o2::base::GRPGeomHelper::instance().getOrbitResetTimeMS();
    const auto firstTFOrbit = pc.services().get<o2::framework::TimingInfo>().firstTForbit;
    const ULong64_t timestamp = orbitResetTimeMS + firstTFOrbit * o2::constants::lhc::LHCOrbitMUS * 0.001;
    mPTHelper.sendPTForTS(pc, timestamp);

    if (mStreamer) {
      const float pressure = mPTHelper.getPressure(timestamp);
      const auto temp = mPTHelper.getTemperature(timestamp);
      (*mStreamer) << "pt"
                   << "pressure=" << pressure
                   << "temperatureA=" << temp.first
                   << "temperatureC=" << temp.second
                   << "time=" << timestamp
                   << "\n";
    }
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
    mPTHelper.accountCCDBInputs(matcher, obj);
  }

 private:
  PressureTemperatureHelper mPTHelper;
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;     ///< info for CCDB request
  std::unique_ptr<o2::utils::TreeStreamRedirector> mStreamer; ///< debug streamer
};

o2::framework::DataProcessorSpec getTPCPressureTemperatureSpec()
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;
  o2::header::DataDescription dataDescription;

  PressureTemperatureHelper::requestCCDBInputs(inputs);
  PressureTemperatureHelper::setOutputs(outputs);

  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                false,                          // GRPECS=true for nHBF per TF
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);
  return DataProcessorSpec{
    "tpc-pressure-temperature",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<PressureTemperatureDevice>(ccdbRequest)},
    Options{
      {"enable-root-output", VariantType::Bool, false, {"Enable root-files output writers"}},
      {"fit-interval", VariantType::Int, 300, {"interval in seconds for which to e.g. perform fits of the temperature sensors"}}} // end Options
  };
}

} // end namespace tpc
} // end namespace o2
