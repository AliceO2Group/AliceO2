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

#include "TPCCalibration/CorrectionMapsLoader.h"
#include "Framework/Logger.h"
#include "Framework/ProcessingContext.h"
#include "Framework/InputRecord.h"
#include "Framework/ConfigParamSpec.h"
#include "TPCFastTransformPOD.h"
using namespace o2::tpc;
using namespace o2::framework;

#ifndef GPUCA_GPUCODE_DEVICE

//________________________________________________________
void CorrectionMapsLoader::extractCCDBInputs(ProcessingContext& pc)
{
  const bool lumiValid = pc.inputs().isValid("lumiCTP");
  if(lumiValid) {
    mInstLumiCTP = pc.inputs().get<float>("lumiCTP");
  }

  const bool mapValid = pc.inputs().isValid("corrMap");
  if(!mapValid) {
    LOGP(info, "No correction map found in the input record!");
    return;
  }

  // get the raw buffer and reinterpret as TPCFastTransformPOD
  auto const& raw = pc.inputs().get<const char*>("corrMap");
  setCorrMap(&gpu::TPCFastTransformPOD::get(raw));
  setUpdatedMap();
}

//________________________________________________________
void CorrectionMapsLoader::requestInputs(std::vector<InputSpec>& inputs, std::vector<o2::framework::ConfigParamSpec>& options)
{
    addInput(inputs, {"corrMap", o2::header::gDataOriginTPC, "TPCCORRMAP", 0, Lifetime::Timeframe});
    addInput(inputs, {"lumiCTP", o2::header::gDataOriginCTP, "LUMICTP", 0, Lifetime::Timeframe});
}

void CorrectionMapsLoader::addInput(std::vector<InputSpec>& inputs, InputSpec&& isp)
{
  if (std::find(inputs.begin(), inputs.end(), isp) == inputs.end()) {
    inputs.emplace_back(isp);
  }
}

// CorrectionMapsLoaderGloOpts CorrectionMapsLoader::parseGlobalOptions(const o2::framework::ConfigParamRegistry& opts)
// {
//   CorrectionMapsLoaderGloOpts tpcopt;
//   auto lumiTypeVal = opts.get<int>("lumi-type");
//   if (lumiTypeVal < -1 || lumiTypeVal > 2) {
//     LOGP(fatal, "Invalid lumi-type value: {}", lumiTypeVal);
//   }
//   tpcopt.lumiType = static_cast<LumiScaleType>(lumiTypeVal);

//   auto lumiModeVal = opts.get<int>("corrmap-lumi-mode");
//   if (lumiModeVal < -1 || lumiModeVal > 2) {
//     LOGP(fatal, "Invalid corrmap-lumi-mode value: {}", lumiModeVal);
//   }
//   tpcopt.lumiMode = static_cast<LumiScaleMode>(lumiModeVal);

//   tpcopt.enableMShapeCorrection = opts.get<bool>("enable-M-shape-correction");
//   tpcopt.requestCTPLumi = !opts.get<bool>("disable-ctp-lumi-request");
//   tpcopt.checkCTPIDCconsistency = !opts.get<bool>("disable-lumi-type-consistency-check");
//   if (!tpcopt.requestCTPLumi && tpcopt.lumiType == LumiScaleType::CTPLumi) {
//     LOGP(fatal, "Scaling with CTP Lumi is requested but this input is disabled");
//   }
//   return tpcopt;
// }

void CorrectionMapsLoader::addGlobalOptions(std::vector<ConfigParamSpec>& options)
{
  // these are options which should be added at the workflow level, since they modify the inputs of the devices
  addOption(options, ConfigParamSpec{"lumi-type", o2::framework::VariantType::Int, 0, {"1 = use CTP lumi for TPC correction scaling, 2 = use TPC scalers for TPC correction scaling"}});
  addOption(options, ConfigParamSpec{"corrmap-lumi-mode", o2::framework::VariantType::Int, 0, {"scaling mode: (default) 0 = static + scale * full; 1 = full + scale * derivative; 2 = full + scale * derivative (for MC)"}});
  addOption(options, ConfigParamSpec{"enable-M-shape-correction", o2::framework::VariantType::Bool, false, {"Enable M-shape distortion correction"}});
  addOption(options, ConfigParamSpec{"disable-ctp-lumi-request", o2::framework::VariantType::Bool, false, {"do not request CTP lumi (regardless what is used for corrections)"}});
  addOption(options, ConfigParamSpec{"disable-lumi-type-consistency-check", o2::framework::VariantType::Bool, false, {"disable check of selected CTP or IDC scaling source being consistent with the map"}});
}

void CorrectionMapsLoader::addOption(std::vector<ConfigParamSpec>& options, ConfigParamSpec&& osp)
{
  if (std::find(options.begin(), options.end(), osp) == options.end()) {
    options.emplace_back(osp);
  }
}

#endif // #ifndef GPUCA_GPUCODE_DEVICE
