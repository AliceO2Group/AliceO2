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
#include "Framework/CallbacksPolicy.h"
#include "Framework/CallbackService.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessingHeader.h"
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
  const auto& hbfu = o2::raw::HBFUtils::Instance();
  for (auto& spec : wf) {
    if (spec.inputs.empty()) {
      o2f::ConfigParamsHelper::addOptionIfMissing(spec.options, o2f::ConfigParamSpec{HBFConfOpt, o2f::VariantType::String, std::string(o2::base::NameConf::DIGITIZATIONCONFIGFILE), {"configKeyValues ini file for HBFUtils, root file with per-TF info or none"}});
    }
  }
}

//_________________________________________________________
HBFUtilsInitializer::HBFOpt HBFUtilsInitializer::getOptType(const std::string& optString)
{
  // return type of the file provided via HBFConfOpt
  HBFOpt opt = HBFOpt::NONE;
  if (!optString.empty()) {
    if (o2::utils::Str::endsWith(optString, ".ini")) {
      opt = HBFOpt::INI;
    } else if (o2::utils::Str::endsWith(optString, ".json")) {
      opt = HBFOpt::JSON;
    } else if (o2::utils::Str::endsWith(optString, ".root")) {
      opt = HBFOpt::ROOT;
    } else if (optString != "none") {
      throw std::runtime_error(fmt::format("invalid option {} for {}", optString, HBFConfOpt));
    }
  }
  return opt;
}

//_________________________________________________________
std::vector<o2::dataformats::TFIDInfo> HBFUtilsInitializer::readTFIDInfoVector(const std::string& fname)
{
  TFile fl(fname.c_str());
  auto vptr = (std::vector<o2::dataformats::TFIDInfo>*)fl.GetObjectChecked("tfidinfo", "std::vector<o2::dataformats::TFIDInfo>");
  if (!vptr) {
    throw std::runtime_error(fmt::format("Failed to read tfidinfo vector from {}", fname));
  }
  std::vector<o2::dataformats::TFIDInfo> v(*vptr);
  return v;
}

//_________________________________________________________
void HBFUtilsInitializer::assignDataHeader(const std::vector<o2::dataformats::TFIDInfo>& tfinfoVec, o2::header::DataHeader& dh)
{
  const auto tfinf = tfinfoVec[dh.tfCounter % tfinfoVec.size()];
  LOGP(DEBUG, "Setting DH for {}/{} from tfCounter={} firstTForbit={} runNumber={} to tfCounter={} firstTForbit={} runNumber={}",
       dh.dataOrigin.as<std::string>(), dh.dataDescription.as<std::string>(), dh.tfCounter, dh.firstTForbit, dh.runNumber, tfinf.tfCounter, tfinf.firstTForbit, tfinf.runNumber);
  dh.firstTForbit = tfinf.firstTForbit;
  dh.tfCounter = tfinf.tfCounter;
  dh.runNumber = tfinf.runNumber;
}

//_________________________________________________________
void HBFUtilsInitializer::addNewTimeSliceCallback(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  policies.push_back(o2::framework::CallbacksPolicy{
    [](o2::framework::DeviceSpec const& spec, o2::framework::ConfigContext const& context) -> bool {
      return (!context.helpOnCommandLine() && o2f::ConfigParamsHelper::hasOption(spec.options, HBFConfOpt));
    },
    [](o2::framework::CallbackService& service, o2::framework::InitContext& context) {
      auto fname = context.options().get<std::string>(HBFConfOpt);
      auto optType = getOptType(fname);
      if (optType == HBFOpt::NONE) {
        return; // no callback
      }
      if (!o2::utils::Str::pathExists(fname)) {
        throw std::runtime_error(fmt::format("file {} does not exist", fname));
      }
      if (optType == HBFOpt::ROOT) { // TF-dependent custom values should be set
        service.set(o2::framework::CallbackService::Id::NewTimeslice,
                    [tfidinfo = readTFIDInfoVector(fname)](o2::header::DataHeader& dh, o2::framework::DataProcessingHeader&) { assignDataHeader(tfidinfo, dh); });
      } else {                                                                // simple linear enumeration
        o2::conf::ConfigurableParam::updateFromFile(fname, "HBFUtils", true); // update only those values which were not touched yet (provenance == kCODE)
        const auto& hbfu = o2::raw::HBFUtils::Instance();
        service.set(o2::framework::CallbackService::Id::NewTimeslice,
                    [offset = int64_t(hbfu.getFirstIRofTF({0, hbfu.orbitFirstSampled}).orbit),
                     increment = int64_t(hbfu.nHBFPerTF)](o2::header::DataHeader& dh, o2::framework::DataProcessingHeader&) {
                      dh.firstTForbit = offset + increment * dh.tfCounter;
                    });
      }
    }});
}
