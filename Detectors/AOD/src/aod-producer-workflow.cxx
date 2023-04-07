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

#include "AODProducerWorkflow/AODProducerWorkflowSpec.h"
#include "Framework/CompletionPolicy.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "CommonUtils/ConfigurableParam.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "DetectorsBase/DPLWorkflowUtils.h"

using namespace o2::framework;
using GID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writer"}},
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation"}},
    {"disable-secondary-vertices", o2::framework::VariantType::Bool, false, {"disable filling secondary vertices"}},
    {"enable-strangeness-tracking", o2::framework::VariantType::Bool, false, {"enable strangeness tracking"}},
    {"info-sources", VariantType::String, std::string{GID::ALL}, {"comma-separated list of sources to use"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}},
    {"combine-source-devices", o2::framework::VariantType::Bool, false, {"merge DPL source devices"}},
    {"ctpconfig-per-run", o2::framework::VariantType::Bool, false, {"Use CTP config per run. 1 -- on (Data), 0 -- off (MC)"}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  auto useMC = !configcontext.options().get<bool>("disable-mc");
  bool enableSV = !configcontext.options().get<bool>("disable-secondary-vertices");
  bool enableST = configcontext.options().get<bool>("enable-strangeness-tracking");
  bool ctpcfgperrun = configcontext.options().get<bool>("ctpconfig-per-run");

  GID::mask_t allowedSrc = GID::getSourcesMask("ITS,MFT,MCH,MID,MCH-MID,TPC,TRD,ITS-TPC,TPC-TOF,TPC-TRD,ITS-TPC-TOF,ITS-TPC-TRD,TPC-TRD-TOF,ITS-TPC-TRD-TOF,MFT-MCH,FT0,FV0,FDD,ZDC,EMC,CTP,PHS,CPV");
  GID::mask_t src = allowedSrc & GID::getSourcesMask(configcontext.options().get<std::string>("info-sources"));

  // manually add TOF to MC mask for addInputSpecs()
  if (src[GID::TPCTOF] || src[GID::ITSTPCTRDTOF] || src[GID::ITSTPCTOF] || src[GID::TPCTRDTOF]) {
    src.set(o2::detectors::DetID::TOF);
  }

  WorkflowSpec specs;
  specs.emplace_back(o2::aodproducer::getAODProducerWorkflowSpec(src, enableSV, enableST, useMC, ctpcfgperrun));

  auto srcCls = src & ~(GID::getSourceMask(GID::MCH) | GID::getSourceMask(GID::MID)); // Don't read global MID and MCH clusters (those attached to tracks are always read)
  auto srcMtc = src;

  WorkflowSpec inputspecs;
  o2::globaltracking::InputHelper::addInputSpecs(configcontext, inputspecs, srcCls, srcMtc, src, useMC, src);
  o2::globaltracking::InputHelper::addInputSpecsPVertex(configcontext, inputspecs, useMC);
  if (enableSV) {
    o2::globaltracking::InputHelper::addInputSpecsSVertex(configcontext, inputspecs);
  }
  if (enableST) {
    o2::globaltracking::InputHelper::addInputSpecsStrangeTrack(configcontext, inputspecs, useMC);
  }
  if (configcontext.options().get<bool>("combine-source-devices")) {
    std::vector<DataProcessorSpec> unmerged;
    specs.push_back(specCombiner("AOD-input-reader", inputspecs, unmerged));
    for (auto& is : unmerged) {
      specs.push_back(is);
    }
  } else {
    for (auto& s : inputspecs) {
      specs.push_back(s);
    }
  }

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  return std::move(specs);
}
