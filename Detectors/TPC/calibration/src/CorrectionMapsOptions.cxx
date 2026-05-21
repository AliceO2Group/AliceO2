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

#include "TPCCalibration/CorrectionMapsOptions.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ConfigParamRegistry.h"
using namespace o2::tpc;
using namespace o2::framework;

//________________________________________________________
CorrectionMapsGloOpts CorrectionMapsOptions::parseGlobalOptions(const o2::framework::ConfigParamRegistry& opts)
{
  CorrectionMapsGloOpts tpcopt;
  auto lumiTypeVal = opts.get<int>("lumi-type");
  if (lumiTypeVal < static_cast<int>(LumiScaleType::Unset) || lumiTypeVal >= static_cast<int>(LumiScaleType::Count)) {
    LOGP(fatal, "Invalid lumi-type value: {}", lumiTypeVal);
  }
  tpcopt.lumiType = static_cast<LumiScaleType>(lumiTypeVal);

  auto lumiModeVal = opts.get<int>("corrmap-lumi-mode");
  if (lumiModeVal < static_cast<int>(LumiScaleMode::Unset) || lumiModeVal >= static_cast<int>(LumiScaleMode::Count)) {
    LOGP(fatal, "Invalid corrmap-lumi-mode value: {}", lumiModeVal);
  }
  tpcopt.lumiMode = static_cast<LumiScaleMode>(lumiModeVal);

  tpcopt.enableMShapeCorrection = opts.get<bool>("enable-M-shape-correction");
  tpcopt.requestCTPLumi = !opts.get<bool>("disable-ctp-lumi-request");
  tpcopt.checkCTPIDCconsistency = !opts.get<bool>("disable-lumi-type-consistency-check");
  if (!tpcopt.requestCTPLumi && tpcopt.lumiType == LumiScaleType::CTPLumi) {
    LOGP(fatal, "Scaling with CTP Lumi is requested but this input is disabled");
  }
  return tpcopt;
}

void CorrectionMapsOptions::addGlobalOptions(std::vector<ConfigParamSpec>& options)
{
  // these are options which should be added at the workflow level, since they modify the inputs of the devices
  addOption(options, ConfigParamSpec{"lumi-type", o2::framework::VariantType::Int, 0, {"1 = use CTP lumi for TPC correction scaling, 2 = use TPC scalers for TPC correction scaling"}});
  addOption(options, ConfigParamSpec{"corrmap-lumi-mode", o2::framework::VariantType::Int, 0, {"scaling mode: (default) 0 = static + scale * full; 1 = full + scale * derivative; 2 = full + scale * derivative (for MC); 3 = no correction; 4 = static only"}});
  addOption(options, ConfigParamSpec{"enable-M-shape-correction", o2::framework::VariantType::Bool, false, {"Enable M-shape distortion correction"}});
  addOption(options, ConfigParamSpec{"disable-ctp-lumi-request", o2::framework::VariantType::Bool, false, {"do not request CTP lumi (regardless what is used for corrections)"}});
  addOption(options, ConfigParamSpec{"disable-lumi-type-consistency-check", o2::framework::VariantType::Bool, false, {"disable check of selected CTP or IDC scaling source being consistent with the map"}});
}

void CorrectionMapsOptions::addOption(std::vector<ConfigParamSpec>& options, ConfigParamSpec&& osp)
{
  if (std::find(options.begin(), options.end(), osp) == options.end()) {
    options.emplace_back(osp);
  }
}
