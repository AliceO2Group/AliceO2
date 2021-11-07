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

// @brief Aux.class initialize HBFUtils
// @author ruben.shahoyan@cern.ch

#include "Headers/DataHeader.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "DetectorsRaw/HBFUtils.h"
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/StringUtils.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "Framework/ConfigContext.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ConfigParamsHelper.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DeviceSpec.h"
#include "Framework/Logger.h"
#include <TFile.h>

using namespace o2::raw;
namespace o2f = o2::framework;

/// If the workflow has devices w/o inputs, we assume that these are data readers in root-file based workflow.
/// In this case this class will configure these devices DataHeader.firstTForbit generator to provide orbit according to HBFUtil setings
/// In case the configcontext has relevant option, the HBFUtils will be beforehand updated from the file indicated by this option.
/// (only those fields of HBFUtils which were not modified before, e.g. by ConfigurableParam::updateFromString)

//_________________________________________________________
HBFUtilsInitializer::HBFUtilsInitializer(const o2f::ConfigContext& configcontext, o2f::WorkflowSpec& wf)
{
  auto updateHBFUtils = [&configcontext]() {
    static bool done = false;
    if (!done) {
      bool helpasked = configcontext.helpOnCommandLine(); // if help is asked, don't take for granted that the ini file is there, don't produce an error if it is not!
      std::string conf = configcontext.options().isSet(HBFConfOpt) ? configcontext.options().get<std::string>(HBFConfOpt) : "";
      if (!helpasked) {
	auto optType = getOptType(conf);
	if (optType == HBFOpt::INI || optType == HBFOpt::JSON) {
	  o2::conf::ConfigurableParam::updateFromFile(conf, "HBFUtils", true); // update only those values which were not touched yet (provenance == kCODE)
	}
      }
      const auto& hbfu = o2::raw::HBFUtils::Instance();
      hbfu.checkConsistency();
      done = true;
    }
  };

  const auto& hbfu = o2::raw::HBFUtils::Instance();
  for (auto& spec : wf) {
    if (spec.inputs.empty()) {
      updateHBFUtils();
      o2f::ConfigParamsHelper::addOptionIfMissing(spec.options, o2f::ConfigParamSpec{"orbit-offset-enumeration", o2f::VariantType::Int64, int64_t(hbfu.getFirstIRofTF({0, hbfu.orbitFirstSampled}).orbit), {"1st injected orbit"}});
      o2f::ConfigParamsHelper::addOptionIfMissing(spec.options, o2f::ConfigParamSpec{"orbit-multiplier-enumeration", o2f::VariantType::Int64, int64_t(hbfu.nHBFPerTF), {"orbits/TF"}});
    }
  }
}

//_________________________________________________________
void HBFUtilsInitializer::addConfigOption(std::vector<o2f::ConfigParamSpec>& opts)
{
  o2f::ConfigParamsHelper::addOptionIfMissing(opts, o2f::ConfigParamSpec{HBFConfOpt, o2f::VariantType::String, std::string(o2::base::NameConf::DIGITIZATIONCONFIGFILE), {"configKeyValues file for HBFUtils, root file with per-TF info or none"}});
}

//_________________________________________________________
HBFUtilsInitializer::HBFOpt HBFUtilsInitializer::getOptType(const std::string& optString)
{
  // return type of the file provided via HBFConfOpt
  HBFOpt opt = HBFOpt::NONE;
  if (!optString.empty()) {
    if (o2::utils::Str::endsWith(optString,".ini")) {
      opt = HBFOpt::INI;
    } else if (o2::utils::Str::endsWith(optString,".json")) {
      opt = HBFOpt::JSON;
    } else if (o2::utils::Str::endsWith(optString,".root")) {
      opt = HBFOpt::ROOT;
    } else if (optString!="none") {
      throw std::runtime_error(fmt::format("invalid option {} for {}", optString, HBFConfOpt));
    }
  }
  if (opt != HBFOpt::NONE && !o2::utils::Str::pathExists(optString)) {
    throw std::runtime_error(fmt::format("file {} does not exist", optString));
  }
  return opt;
}

//_________________________________________________________
bool HBFUtilsInitializer::needToAttachDataHeaderCallBack(const o2::framework::DeviceSpec& spec, const o2::framework::ConfigContext& context)
{
  // decide if DataHeader callback need to be attached to device  
  bool hasRealInput = false;
  for (auto& inp :  spec.inputChannels) {
    if (!o2::utils::Str::beginsWith(inp.name,"from_internal-dpl-clock_to")) {
      hasRealInput = true;
      break;
    }
  }
  return (!context.helpOnCommandLine()) && !hasRealInput && (spec.name!="internal-dpl-clock") && getOptType(context.options().get<std::string>(HBFConfOpt))==HBFOpt::ROOT;
}

//_________________________________________________________
std::vector<o2::dataformats::TFIDInfo> HBFUtilsInitializer::readTFIDInfoVector(const std::string& fname)
{
  TFile fl(fname.c_str());
  auto vptr = (std::vector<o2::dataformats::TFIDInfo>*)fl.GetObjectChecked("tfidinfo","std::vector<o2::dataformats::TFIDInfo>");
  if (!vptr) {
    throw std::runtime_error(fmt::format("Failed to read tfidinfo vector from {}", fname));
  }
  std::vector<o2::dataformats::TFIDInfo> v(*vptr);
  return v;
}

//_________________________________________________________
void HBFUtilsInitializer::assignDataHeader(const std::vector<o2::dataformats::TFIDInfo>& tfinfoVec, o2::header::DataHeader& dh)
{
  const auto tfinf = tfinfoVec[dh.tfCounter%tfinfoVec.size()];
  LOGP(DEBUG, "Setting DH for {}/{} from tfCounter={} firstTForbit={} runNumber={} to tfCounter={} firstTForbit={} runNumber={}",
       dh.dataOrigin.as<std::string>(), dh.dataDescription.as<std::string>(),dh.tfCounter, dh.firstTForbit, dh.runNumber, tfinf.tfCounter, tfinf.firstTForbit, tfinf.runNumber);
  dh.firstTForbit = tfinf.firstTForbit;
  dh.tfCounter = tfinf.tfCounter;
  dh.runNumber = tfinf.runNumber;
}
