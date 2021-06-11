// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// @brief Aux.class initialize HBFUtils
// @author ruben.shahoyan@cern.ch

#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "DetectorsRaw/HBFUtils.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "Framework/ConfigContext.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ConfigParamsHelper.h"
#include "Framework/Logger.h"

using namespace o2::raw;
namespace o2f = o2::framework;

/// If the workflow has devices w/o inputs, we assume that these are data readers in root-file based workflow.
/// In this case this class will configure these devices DataHeader.firstTForbit generator to provide orbit according to HBFUtil setings
/// In case the configcontext has relevant option, the HBFUtils will be beforehand updated from the file indicated by this option.
/// (only those fields of HBFUtils which were not modified before, e.g. by ConfigurableParam::updateFromString)

HBFUtilsInitializer::HBFUtilsInitializer(const o2f::ConfigContext& configcontext, o2f::WorkflowSpec& wf)
{
  auto updateHBFUtils = [&configcontext]() {
    static bool done = false;
    if (!done) {
      bool helpasked = configcontext.helpOnCommandLine(); // if help is asked, don't take for granted that the ini file is there, don't produce an error if it is not!
      std::string conf = configcontext.options().isSet(HBFConfOpt) ? configcontext.options().get<std::string>(HBFConfOpt) : "";
      if (!conf.empty() && conf != "none" && !(helpasked && !o2::utils::Str::pathExists(conf))) {
        o2::conf::ConfigurableParam::updateFromFile(conf, "HBFUtils", true); // update only those values which were not touched yet (provenance == kCODE)
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

void HBFUtilsInitializer::addConfigOption(std::vector<o2f::ConfigParamSpec>& opts)
{
  o2f::ConfigParamsHelper::addOptionIfMissing(opts, o2f::ConfigParamSpec{HBFConfOpt, o2f::VariantType::String, std::string(o2::base::NameConf::DIGITIZATIONCONFIGFILE), {"configKeyValues file for HBFUtils (or none)"}});
}
