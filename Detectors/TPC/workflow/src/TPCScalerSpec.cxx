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

/// \file TPCScalerSpec.cxx
/// \brief device for providing tpc scaler per TF
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Nov 4, 2023

#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "TPCBase/CDBInterface.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "TPCCalibration/TPCScaler.h"
#include "TTree.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{

class TPCScalerSpec : public Task
{
 public:
  TPCScalerSpec(std::shared_ptr<o2::base::GRPGeomRequest> req) : mCCDBRequest(req){};

  void init(framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    mIonDriftTimeMS = ic.options().get<float>("ion-drift-time");
  }

  void run(ProcessingContext& pc) final
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    if (pc.inputs().isValid("tpcscaler")) {
      pc.inputs().get<TTree*>("tpcscaler");
    }

    if (pc.services().get<o2::framework::TimingInfo>().runNumber != mTPCScaler.getRun()) {
      LOGP(error, "Run number {} of processed data and run number {} of loaded TPC scaler doesnt match!", pc.services().get<o2::framework::TimingInfo>().runNumber, mTPCScaler.getRun());
    }

    const auto orbitResetTimeMS = o2::base::GRPGeomHelper::instance().getOrbitResetTimeMS();
    const auto firstTFOrbit = pc.services().get<o2::framework::TimingInfo>().firstTForbit;
    const double timestamp = orbitResetTimeMS + firstTFOrbit * o2::constants::lhc::LHCOrbitMUS * 0.001;
    float scalerA = mTPCScaler.getMeanScaler(timestamp, o2::tpc::Side::A);
    float scalerC = mTPCScaler.getMeanScaler(timestamp, o2::tpc::Side::C);
    float meanScaler = (scalerA + scalerC) / 2;
    LOGP(info, "Publishing TPC scaler: {}", meanScaler);
    pc.outputs().snapshot(Output{header::gDataOriginTPC, "TPCSCALER"}, meanScaler);
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
    if (matcher == ConcreteDataMatcher(o2::header::gDataOriginTPC, "TPCSCALERCCDB", 0)) {
      LOGP(info, "Updating TPC scaler");
      mTPCScaler.setFromTree(*((TTree*)obj));
      if (mIonDriftTimeMS > 0) {
        LOGP(info, "Setting ion drift time to: {}", mIonDriftTimeMS);
        mTPCScaler.setIonDriftTimeMS(mIonDriftTimeMS);
      }
    }
  }

 private:
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest; ///< info for CCDB request
  float mIonDriftTimeMS{-1};                              ///< ion drift time
  TPCScaler mTPCScaler;                                   ///< tpc scaler
};

o2::framework::DataProcessorSpec getTPCScalerSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("tpcscaler", o2::header::gDataOriginTPC, "TPCSCALERCCDB", 0, Lifetime::Condition, ccdbParamSpec(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalScaler)));

  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                false,                          // GRPECS=true for nHBF per TF
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTPC, "TPCSCALER", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "tpc-scaler",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCScalerSpec>(ccdbRequest)},
    Options{
      {"ion-drift-time", VariantType::Float, -1.f, {"Overwrite ion drift time if a value >0 is provided"}}}};
}

} // namespace tpc
} // namespace o2
