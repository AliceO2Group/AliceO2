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

/// \file align-record-workflow.cxx
/// \brief Implementation of a DPL device to create MillePede record for muon alignment
///
/// \author Chi ZHANG, CEA-Saclay, chi.zhang@cern.ch

#include "MCHAlign/AlignRecordSpec.h"

#include "Framework/CompletionPolicy.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "CommonUtils/ConfigurableParam.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "ForwardAlign/MilleRecordWriterSpec.h"

using namespace o2::framework;
using namespace std;

using GID = o2::dataformats::GlobalTrackID;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation"}},
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"do not write output root files"}},
    {"disable-ccdb", VariantType::Bool, false, {"disable input files from CCDB"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"
WorkflowSpec defineDataProcessing(const ConfigContext& configcontext)
{
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  o2::conf::ConfigurableParam::writeINI("o2mchalignrecord-workflow_configuration.ini");

  auto useMC = !configcontext.options().get<bool>("disable-mc");
  bool disableCCDB = configcontext.options().get<bool>("disable-ccdb");
  bool disableRootOutput = configcontext.options().get<bool>("disable-root-output");

  WorkflowSpec specs;
  specs.emplace_back(o2::mch::getAlignRecordSpec(useMC, disableCCDB));
  auto srcTracks = GID::getSourcesMask("MCH");
  auto srcClusters = GID::getSourcesMask("MCH");
  auto matchMask = GID::getSourcesMask("MCH-MID");

  if (!disableRootOutput) {
    specs.emplace_back(o2::fwdalign::getMilleRecordWriterSpec(useMC));
  }

  o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, srcClusters, matchMask, srcTracks, useMC, srcClusters, srcTracks);

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  return std::move(specs);
}