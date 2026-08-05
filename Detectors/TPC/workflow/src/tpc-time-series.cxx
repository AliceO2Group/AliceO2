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

/// \file   tpc-time-series.cxx
/// \author Matthias Kleiner, mkleiner@ikf.uni-frankfurt.de

#include "TPCWorkflow/TPCTimeSeriesSpec.h"
#include "TPCWorkflow/TPCTimeSeriesWriterSpec.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/ConfigurableParam.h"
#include "TPCReaderWorkflow/TPCSectorCompletionPolicy.h"
#include "DetectorsBase/DPLWorkflowUtils.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "DataFormatsITSMFT/DPLAlpideParamInitializer.h"
#include "Framework/ConfigParamSpec.h"
#include "GPUDebugStreamer.h"

using namespace o2::framework;
using GID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;

// ------------------------------------------------------------------
void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<ConfigParamSpec> options{
    ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}},
    {"disable-root-output", VariantType::Bool, false, {"disable root-files output writers"}},
    {"disable-root-input", VariantType::Bool, false, {"disable root-files input reader"}},
    {"enable-unbinned-root-output", VariantType::Bool, false, {"writing out unbinned track data"}},
    {"track-sources", VariantType::String, std::string{GID::ALL}, {"comma-separated list of sources to use"}},
    {"material-type", VariantType::Int, 2, {"Type for the material budget during track propagation: 0=None, 1=Geo, 2=LUT"}}};
  o2::itsmft::DPLAlpideParamInitializer::addITSConfigOption(options);
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  WorkflowSpec workflow;
  o2::conf::ConfigurableParam::updateFromString(config.options().get<std::string>("configKeyValues"));
  const bool disableWriter = config.options().get<bool>("disable-root-output");
  const bool enableUnbinnedWriter = config.options().get<bool>("enable-unbinned-root-output");
  GID::mask_t allowedSources = GID::getSourcesMask("ITS,TPC,ITS-TPC,ITS-TPC-TRD,ITS-TPC-TOF,ITS-TPC-TRD-TOF,FT0");
  auto srcTrc = allowedSources & GID::getSourcesMask(config.options().get<std::string>("track-sources"));
  o2::dataformats::GlobalTrackID::mask_t srcCls = GID::getSourcesMask("TPC");
  if (GID::includesDet(DetID::ITS, srcTrc)) {
    srcCls |= GID::getSourcesMask("ITS");
  }
  if (GID::includesDet(DetID::TRD, srcTrc)) {
    srcCls |= GID::getSourcesMask("TRD");
  }
  if (GID::includesDet(DetID::TOF, srcTrc)) {
    srcCls |= GID::getSourcesMask("TOF");
  }

  auto materialType = static_cast<o2::base::Propagator::MatCorrType>(config.options().get<int>("material-type"));

  o2::globaltracking::InputHelper::addInputSpecs(config, workflow, srcCls, srcTrc, srcTrc, false);
  o2::globaltracking::InputHelper::addInputSpecsPVertex(config, workflow, false); // P-vertex is always needed

  workflow.emplace_back(o2::tpc::getTPCTimeSeriesSpec(disableWriter, materialType, enableUnbinnedWriter, srcTrc));
  if (!disableWriter) {
    workflow.emplace_back(o2::tpc::getTPCTimeSeriesWriterSpec());
  }
  o2::raw::HBFUtilsInitializer hbfIni(config, workflow);
  return workflow;
}
