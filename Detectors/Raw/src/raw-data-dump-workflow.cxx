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

#include "CommonUtils/ConfigurableParam.h"
#include "Framework/ConfigParamRegistry.h"
#include "DetectorsRaw/RawDumpSpec.h"

using namespace o2::framework;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options;
  options.push_back(ConfigParamSpec{"onlyDet", VariantType::String, "all", {"list of dectors"}});
  options.push_back(ConfigParamSpec{"tof-input-uncompressed", VariantType::Bool, false, {"TOF input is original (RAWDATA) rather than compressed (CRAWDATA)"}});
  options.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"semicolon separated key=value strings"}});
  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  auto detlistSelect = configcontext.options().get<std::string>("onlyDet");
  auto tofOrig = configcontext.options().get<bool>("tof-input-uncompressed");
  const o2::detectors::DetID::mask_t detMaskFilter = o2::detectors::DetID::getMask("ITS,TPC,TRD,TOF,PHS,CPV,EMC,HMP,MFT,MCH,MID,ZDC,FT0,FV0,FDD,CTP");
  o2::detectors::DetID::mask_t detMask = o2::detectors::DetID::getMask(detlistSelect) & detMaskFilter;
  WorkflowSpec specs{o2::raw::getRawDumpSpec(detMask, tofOrig)};
  return specs;
}
