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

#include "Framework/AnalysisSupportHelpers.h"
#include "Framework/DataOutputDirector.h"
#include "Framework/DataSpecViews.h"
#include "Framework/PluginManager.h"
#include "Framework/ConfigContext.h"
#include "WorkflowHelpers.h"

template class std::vector<o2::framework::OutputObjectInfo>;
template class std::vector<o2::framework::OutputTaskInfo>;

namespace o2::framework
{

std::shared_ptr<DataOutputDirector> AnalysisSupportHelpers::getDataOutputDirector(ConfigContext const& ctx)
{
  auto const& options = ctx.options();
  auto const& OutputsInputs = ctx.services().get<DanglingEdgesContext>().outputsInputs;
  auto const& isDangling = ctx.services().get<DanglingEdgesContext>().isDangling;

  std::shared_ptr<DataOutputDirector> dod = std::make_shared<DataOutputDirector>();

  // analyze options and take actions accordingly
  // default values
  std::string rdn, resdir("./");
  std::string fnb, fnbase("AnalysisResults_trees");
  float mfs, maxfilesize(-1.);
  std::string fmo, filemode("RECREATE");
  int ntfm, ntfmerge = 1;

  // values from json
  if (options.isSet("aod-writer-json")) {
    auto fnjson = options.get<std::string>("aod-writer-json");
    if (!fnjson.empty()) {
      std::tie(rdn, fnb, fmo, mfs, ntfm) = dod->readJson(fnjson);
      if (!rdn.empty()) {
        resdir = rdn;
      }
      if (!fnb.empty()) {
        fnbase = fnb;
      }
      if (!fmo.empty()) {
        filemode = fmo;
      }
      if (mfs > 0.) {
        maxfilesize = mfs;
      }
      if (ntfm > 0) {
        ntfmerge = ntfm;
      }
    }
  }

  // values from command line options, information from json is overwritten
  if (options.isSet("aod-writer-resdir")) {
    rdn = options.get<std::string>("aod-writer-resdir");
    if (!rdn.empty()) {
      resdir = rdn;
    }
  }
  if (options.isSet("aod-writer-resfile")) {
    fnb = options.get<std::string>("aod-writer-resfile");
    if (!fnb.empty()) {
      fnbase = fnb;
    }
  }
  if (options.isSet("aod-writer-resmode")) {
    fmo = options.get<std::string>("aod-writer-resmode");
    if (!fmo.empty()) {
      filemode = fmo;
    }
  }
  if (options.isSet("aod-writer-maxfilesize")) {
    mfs = options.get<float>("aod-writer-maxfilesize");
    if (mfs > 0) {
      maxfilesize = mfs;
    }
  }
  if (options.isSet("aod-writer-ntfmerge")) {
    ntfm = options.get<int>("aod-writer-ntfmerge");
    if (ntfm > 0) {
      ntfmerge = ntfm;
    }
  }
  // parse the keepString
  if (options.isSet("aod-writer-keep")) {
    auto keepString = options.get<std::string>("aod-writer-keep");
    if (!keepString.empty()) {
      dod->reset();
      std::string d("dangling");
      if (keepString.starts_with(d)) {
        // use the dangling outputs
        std::vector<InputSpec> danglingOutputs;
        for (auto ii = 0u; ii < OutputsInputs.size(); ii++) {
          if (DataSpecUtils::partialMatch(OutputsInputs[ii], writableAODOrigins) && isDangling[ii]) {
            danglingOutputs.emplace_back(OutputsInputs[ii]);
          }
        }
        dod->readSpecs(danglingOutputs);
      } else {
        // use the keep string
        dod->readString(keepString);
      }
    }
  }
  dod->setResultDir(resdir);
  dod->setFilenameBase(fnbase);
  dod->setFileMode(filemode);
  dod->setMaximumFileSize(maxfilesize);
  dod->setNumberTimeFramesToMerge(ntfmerge);

  return dod;
}

void AnalysisSupportHelpers::addMissingOutputsToReader(std::vector<OutputSpec> const& providedOutputs,
                                                       std::vector<InputSpec> const& requestedInputs,
                                                       DataProcessorSpec& publisher)
{
  requestedInputs |
    views::filter_not_matching(providedOutputs) |   // filter the inputs that are already provided
    views::filter_not_matching(publisher.outputs) | // filter the inputs that are already covered
    views::input_to_output_specs() |
    sinks::append_to{publisher.outputs}; // append them to the publisher outputs
}

void AnalysisSupportHelpers::addMissingOutputsToSpawner(std::vector<OutputSpec> const& providedSpecials,
                                                        std::vector<InputSpec> const& requestedSpecials,
                                                        std::vector<InputSpec>& requestedAODs,
                                                        DataProcessorSpec& publisher)
{
  requestedSpecials |
    views::filter_not_matching(providedSpecials) | // filter the inputs that are already provided
    views::input_to_output_specs() |
    sinks::append_to{publisher.outputs}; // append them to the publisher outputs

  std::vector<InputSpec> additionalInputs;
  for (auto const& input : requestedSpecials | views::filter_not_matching(providedSpecials)) {
    input.metadata |
      views::filter_string_params_with("input:") |
      views::params_to_input_specs() |
      sinks::update_input_list{additionalInputs}; // store into a temporary
  }
  additionalInputs | sinks::update_input_list{requestedAODs};    // update requestedAODs
  additionalInputs | sinks::update_input_list{publisher.inputs}; // update publisher inputs
}

void AnalysisSupportHelpers::addMissingOutputsToBuilder(std::vector<InputSpec> const& requestedSpecials,
                                                        std::vector<InputSpec>& requestedAODs,
                                                        std::vector<InputSpec>& requestedDYNs,
                                                        DataProcessorSpec& publisher)
{
  requestedSpecials |
    views::input_to_output_specs() |
    sinks::append_to{publisher.outputs}; // append them to the publisher outputs

  std::vector<InputSpec> additionalInputs;
  for (auto const& input : requestedSpecials) {
    input.metadata |
      views::filter_string_params_with("input:") |
      views::params_to_input_specs() |
      sinks::update_input_list{additionalInputs}; // store into a temporary
  }

  additionalInputs | sinks::update_input_list{publisher.inputs}; // update publisher inputs
  // FIXME: until we have a single list of pairs
  additionalInputs |
    views::partial_match_filter(AODOrigins) |
    sinks::update_input_list{requestedAODs}; // update requestedAODs
  additionalInputs |
    views::partial_match_filter(header::DataOrigin{"DYN"}) |
    sinks::update_input_list{requestedDYNs}; // update requestedDYNs
}

// =============================================================================
DataProcessorSpec AnalysisSupportHelpers::getOutputObjHistSink(ConfigContext const& ctx)
{
  // Lifetime is sporadic because we do not ask each analysis task to send its
  // results every timeframe.
  DataProcessorSpec spec{
    .name = "internal-dpl-aod-global-analysis-file-sink",
    .inputs = {InputSpec("x", DataSpecUtils::dataDescriptorMatcherFrom(header::DataOrigin{"ATSK"}), Lifetime::Sporadic)},
    .outputs = {},
    .algorithm = PluginManager::loadAlgorithmFromPlugin("O2FrameworkAnalysisSupport", "ROOTObjWriter", ctx),
  };

  return spec;
}

// add sink for the AODs
DataProcessorSpec
  AnalysisSupportHelpers::getGlobalAODSink(ConfigContext const& ctx)
{
  auto& ac = ctx.services().get<DanglingEdgesContext>();

  // the command line options relevant for the writer are global
  // see runDataProcessing.h
  DataProcessorSpec spec{
    .name = "internal-dpl-aod-writer",
    .inputs = ac.outputsInputsAOD,
    .outputs = {},
    .algorithm = PluginManager::loadAlgorithmFromPlugin("O2FrameworkAnalysisSupport", "ROOTTTreeWriter", ctx),
  };

  return spec;
}
} // namespace o2::framework
