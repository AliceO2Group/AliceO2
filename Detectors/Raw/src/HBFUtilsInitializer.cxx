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
  auto updateHBFUtils = [&configcontext]() -> std::string {
    static bool done = false;
    static std::string confTFInfo{};
    if (!done) {
      bool helpasked = configcontext.helpOnCommandLine(); // if help is asked, don't take for granted that the ini file is there, don't produce an error if it is not!
      auto conf = configcontext.options().isSet(HBFConfOpt) ? configcontext.options().get<std::string>(HBFConfOpt) : "";
      if (!conf.empty()) {
        auto opt = getOptType(conf);
        if ((opt == HBFOpt::INI || opt == HBFOpt::JSON) && (!(helpasked && !o2::conf::ConfigurableParam::configFileExists(conf)))) {
          o2::conf::ConfigurableParam::updateFromFile(conf, "HBFUtils", true); // update only those values which were not touched yet (provenance == kCODE)
          const auto& hbfu = o2::raw::HBFUtils::Instance();
          hbfu.checkConsistency();
          confTFInfo = HBFUSrc;
        } else if (opt == HBFOpt::HBFUTILS) {
          const auto& hbfu = o2::raw::HBFUtils::Instance();
          hbfu.checkConsistency();
          confTFInfo = HBFUSrc;
        } else if (opt == HBFOpt::ROOT) {
          confTFInfo = conf;
        }
      }
      done = true;
    }
    return confTFInfo;
  };

  if (configcontext.options().hasOption("disable-root-input") && configcontext.options().get<bool>("disable-root-input")) {
    return; // we apply HBFUtilsInitializer only in case of root readers
  }

  const auto& hbfu = o2::raw::HBFUtils::Instance();
  for (auto& spec : wf) {
    if (spec.inputs.empty()) {
      auto conf = updateHBFUtils();
      o2f::ConfigParamsHelper::addOptionIfMissing(spec.options, o2f::ConfigParamSpec{HBFTFInfoOpt, o2f::VariantType::String, conf, {"root file with per-TF info"}});
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
    } else if (optString == HBFUSrc) {
      opt = HBFOpt::HBFUTILS;
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
void HBFUtilsInitializer::assignDataHeader(const std::vector<o2::dataformats::TFIDInfo>& tfinfoVec, o2::header::DataHeader& dh, o2::framework::DataProcessingHeader& dph)
{
  const auto tfinf = tfinfoVec[dh.tfCounter % tfinfoVec.size()];
  LOGP(debug, "Setting DH for {}/{} from tfCounter={} firstTForbit={} runNumber={} to tfCounter={} firstTForbit={} runNumber={}",
       dh.dataOrigin.as<std::string>(), dh.dataDescription.as<std::string>(), dh.tfCounter, dh.firstTForbit, dh.runNumber, tfinf.tfCounter, tfinf.firstTForbit, tfinf.runNumber);
  dh.firstTForbit = tfinf.firstTForbit;
  dh.tfCounter = tfinf.tfCounter;
  dh.runNumber = tfinf.runNumber;
  dph.creation = tfinf.creation;
}

//_________________________________________________________
void HBFUtilsInitializer::addNewTimeSliceCallback(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  policies.push_back(o2::framework::CallbacksPolicy{
    [](o2::framework::DeviceSpec const& spec, o2::framework::ConfigContext const& context) -> bool {
      return (!context.helpOnCommandLine() && o2f::ConfigParamsHelper::hasOption(spec.options, HBFTFInfoOpt));
    },
    [](o2::framework::CallbackService& service, o2::framework::InitContext& context) {
      auto fname = context.options().get<std::string>(HBFTFInfoOpt);
      if (!fname.empty()) {
        if (fname == HBFUSrc) { // simple linear enumeration from already updated HBFUtils
          const auto& hbfu = o2::raw::HBFUtils::Instance();
          service.set(o2::framework::CallbackService::Id::NewTimeslice,
                      [offset = int64_t(hbfu.getFirstIRofTF({0, hbfu.orbitFirstSampled}).orbit), increment = int64_t(hbfu.nHBFPerTF),
                       startTime = hbfu.startTime, orbitFirst = hbfu.orbitFirst, runNumber = hbfu.runNumber](o2::header::DataHeader& dh, o2::framework::DataProcessingHeader& dph) {
                        dh.firstTForbit = offset + increment * dh.tfCounter;
                        dh.runNumber = runNumber;
                        dph.creation = startTime + (dh.firstTForbit - orbitFirst) * o2::constants::lhc::LHCOrbitMUS * 1.e-3;
                      });
        } else if (o2::utils::Str::endsWith(fname, ".root")) { // read TFIDinfo from file
          if (!o2::utils::Str::pathExists(fname)) {
            throw std::runtime_error(fmt::format("file {} does not exist", fname));
          }
          service.set(o2::framework::CallbackService::Id::NewTimeslice,
                      [tfidinfo = readTFIDInfoVector(fname)](o2::header::DataHeader& dh, o2::framework::DataProcessingHeader& dph) { assignDataHeader(tfidinfo, dh, dph); });
        } else { // do not modify timing info
          // we may remove the highest bit set on the creation time?
        }
      }
    }});
}

void HBFUtilsInitializer::addConfigOption(std::vector<o2f::ConfigParamSpec>& opts, const std::string& defOpt)
{
  o2f::ConfigParamsHelper::addOptionIfMissing(opts, o2f::ConfigParamSpec{HBFConfOpt, o2f::VariantType::String, defOpt, {R"(ConfigurableParam ini file or "hbfutils" for HBFUtils, root file with per-TF info or "none")"}});
}
