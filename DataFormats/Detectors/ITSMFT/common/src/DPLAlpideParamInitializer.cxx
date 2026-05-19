// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsITSMFT/DPLAlpideParamInitializer.h"
#include "Framework/ConfigParamsHelper.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ConfigContext.h"

namespace o2::itsmft
{

void DPLAlpideParamInitializer::addConfigOption(std::vector<o2::framework::ConfigParamSpec>& opts)
{
  addITSConfigOption(opts);
  addMFTConfigOption(opts);
}

void DPLAlpideParamInitializer::addITSConfigOption(std::vector<o2::framework::ConfigParamSpec>& opts)
{
  o2::framework::ConfigParamsHelper::addOptionIfMissing(opts, {stagITSOpt, o2::framework::VariantType::Bool, stagDef, {"enable per layer ITS in&out-put for staggered readout"}});
}

void DPLAlpideParamInitializer::addMFTConfigOption(std::vector<o2::framework::ConfigParamSpec>& opts)
{
  o2::framework::ConfigParamsHelper::addOptionIfMissing(opts, {stagMFTOpt, o2::framework::VariantType::Bool, stagDef, {"enable per layer MFT in&out-put for staggered readout"}});
}

bool DPLAlpideParamInitializer::isITSStaggeringEnabled(const o2::framework::ConfigContext& cfgc)
{
  return cfgc.options().get<bool>(stagITSOpt);
}

bool DPLAlpideParamInitializer::isMFTStaggeringEnabled(const o2::framework::ConfigContext& cfgc)
{
  return cfgc.options().get<bool>(stagMFTOpt);
}

} // namespace o2::itsmft
