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

/// @file   CcdbRequestSpec.cxx

#include <vector>

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "DataFormatsParameters/GRPObject.h"
#include "CommonUtils/NameConf.h"
#include "TOFWorkflowIO/CcdbRequestSpec.h"
#include "TOFBase/Utils.h"

using namespace o2::framework;
using namespace o2::tof;

namespace o2
{
namespace tof
{

void CcdbRequest::init(InitContext& ic)
{
  LOG(debug) << "Init TOF CCDB Request!";
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
}

void CcdbRequest::run(ProcessingContext& pc)
{
  static uint32_t counts = 0;
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool firstCall = true;
  if (firstCall) {
    Utils::setNOrbitInTF(o2::base::GRPGeomHelper::instance().getGRPECS()->getNHBFPerTF());
    LOG(info) << "NHBFperTF = " << Utils::getNOrbitInTF();
  }
  counts++;
  LOG(debug) << counts << ") NHBFperTF = " << Utils::getNOrbitInTF();

  firstCall = false;
}

DataProcessorSpec getCcdbRequestSpec()
{
  std::vector<OutputSpec> outputs;
  std::vector<InputSpec> inputs;

  inputs.emplace_back("stdDist", "FLP", "DISTSUBTIMEFRAME", 0, Lifetime::Timeframe);

  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                          // orbitResetTime
                                                                true,                           // GRPECS=true for nHBF per TF
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);

  return DataProcessorSpec{
    "tof-ccdb-request",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<CcdbRequest>(ccdbRequest)},
    Options{}};
}

} // namespace tof
} // namespace o2
