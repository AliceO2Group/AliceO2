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

#include "MFTWorkflow/CAWorkflowOptions.h"

#include <stdexcept>
#include "CommonUtils/ConfigurableParam.h"
#include "DataFormatsITSMFT/DPLAlpideParamInitializer.h"
#include "Framework/ConfigContext.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "ITSMFTTracking/TrackingConfigParam.h"
#include "MFTTracking/MFTTrackingParam.h"

namespace o2::mft::ca
{
WorkflowOptions resolveWorkflowOptions(const WorkflowOptionInput& input, const TrackerOptionAliases& aliases)
{
  using namespace o2::itsmft;
  if (input.upstreamDigits && input.upstreamClusters) {
    throw std::invalid_argument("--digits-from-upstream conflicts with --clusters-from-upstream; choose one input stage");
  }
  if (input.mode < TrackingMode::Unset || input.mode > TrackingMode::Off ||
      aliases.mode < TrackingMode::Unset || aliases.mode > TrackingMode::Off) {
    throw std::invalid_argument("Invalid --tracking-mode or MFTCATrackerParam.trackingMode");
  }
  if (input.nThreads <= 0 || aliases.nThreads <= 0) {
    throw std::invalid_argument("--nThreads and MFTCATrackerParam.nThreads must both be > 0");
  }
  if (!input.runTracking && (input.assessment || input.tracksToRecords)) {
    throw std::invalid_argument("--disable-tracking conflicts with --run-assessment/--run-tracks2records");
  }
  WorkflowOptions result;
  result.kind = input.kind;
  result.input = input.kind == WorkflowKind::TrackerOnly || input.upstreamClusters ? InputStage::UpstreamClusters
                 : input.upstreamDigits                                            ? InputStage::UpstreamDigits
                                                                                   : InputStage::DigitsFile;
  result.output = input.disableRootOutput ? (input.clusterROFsOnly ? OutputPolicy::ClusterROFs : OutputPolicy::None)
                  : input.clusterROFsOnly ? OutputPolicy::TracksAndClusterROFs
                                          : OutputPolicy::All;
  result.tracker.useMC = input.useMC;
  result.tracker.geometry = input.fullGeometry ? GeometrySource::Full : GeometrySource::MFT;
  result.tracker.mode = aliases.mode == TrackingMode::Unset ? input.mode : static_cast<TrackingMode::Type>(aliases.mode);
  if (result.tracker.mode == TrackingMode::Unset) {
    result.tracker.mode = TrackingMode::Sync;
  }
  result.tracker.nThreads = aliases.nThreads;
  result.tracker.filterIRFrames = aliases.filterIRFrames;
  if (input.useIRFrames || aliases.filterIRFrames) {
    result.tracker.irFrames = result.input == InputStage::DigitsFile ? IRFrameSource::File : IRFrameSource::Upstream;
  }
  result.staggering = input.staggering;
  result.runTracking = input.runTracking;
  result.assessment = input.assessment;
  result.processGenerated = input.processGenerated;
  result.tracksToRecords = input.tracksToRecords;
  if (aliases.mode != TrackingMode::Unset && input.mode != result.tracker.mode) {
    result.diagnostics.push_back("MFTCATrackerParam.trackingMode=" + TrackingMode::toString(result.tracker.mode) +
                                 " overrides --tracking-mode=" + TrackingMode::toString(input.mode));
  }
  if (input.nThreads != aliases.nThreads) {
    result.diagnostics.push_back("MFTCATrackerParam.nThreads=" + std::to_string(aliases.nThreads) +
                                 " overrides --nThreads=" + std::to_string(input.nThreads));
  }
  if (!input.runTracking && (input.useIRFrames || aliases.filterIRFrames)) {
    result.diagnostics.push_back("--disable-tracking: --use-irframes/MFTTrackingParam.irFramesOnly have no tracker consumer");
    result.tracker.irFrames = IRFrameSource::None;
  }
  if (input.clusterROFsOnly && input.disableRootOutput) {
    result.diagnostics.push_back("--cluster-rof-branch-only overrides --disable-root-output for the cluster ROF branch");
  }
  return result;
}

WorkflowOptions readWorkflowOptions(const o2::framework::ConfigContext& context, WorkflowKind kind)
{
  const auto& options = context.options();
  using Param = o2::itsmft::TrackerParamConfig<o2::detectors::DetID::MFT>;
  (void)Param::Instance();
  WorkflowOptionInput input;
  input.kind = kind;
  input.nThreads = options.get<int>("nThreads");
  // Apply the CLI alias first, then let explicit parameter keys override it.
  o2::conf::ConfigurableParam::setValue<int>("MFTCATrackerParam", "nThreads", input.nThreads);
  o2::conf::ConfigurableParam::updateFromString(options.get<std::string>("configKeyValues"));
  input.mode = o2::itsmft::TrackingMode::fromString(options.get<std::string>("tracking-mode"));
  input.useMC = !options.get<bool>("disable-mc");
  input.disableRootOutput = options.get<bool>("disable-root-output");
  input.fullGeometry = options.get<bool>("use-geom") || options.get<bool>("use-full-geometry");

  input.useIRFrames = options.get<bool>("use-irframes");
  if (kind == WorkflowKind::Reconstruction) {
    input.upstreamDigits = options.get<bool>("digits-from-upstream");
    input.upstreamClusters = options.get<bool>("clusters-from-upstream");
    input.clusterROFsOnly = options.get<bool>("cluster-rof-branch-only");
    input.runTracking = !options.get<bool>("disable-tracking");
    input.assessment = options.get<bool>("run-assessment");
    input.processGenerated = !options.get<bool>("disable-process-gen");
    input.tracksToRecords = options.get<bool>("run-tracks2records");
    input.staggering = o2::itsmft::DPLAlpideParamInitializer::isMFTStaggeringEnabled(context);
  }
  o2::itsmft::TrackingMode::validateCommonCAOptions(o2::detectors::DetID::MFT);
  const auto& params = Param::Instance();
  auto result = resolveWorkflowOptions(input, {params.trackingMode, params.nThreads, MFTTrackingParam::Instance().irFramesOnly});
  for (const auto& diagnostic : result.diagnostics) {
    LOGP(info, "{}", diagnostic);
  }
  const auto inputName = result.input == InputStage::DigitsFile ? "digits file" : result.input == InputStage::UpstreamDigits ? "upstream digits"
                                                                                                                             : "upstream clusters+patterns+ROFs (and labels if MC enabled)";
  const auto irName = result.tracker.irFrames == IRFrameSource::None ? "none" : result.tracker.irFrames == IRFrameSource::File ? "file"
                                                                                                                               : "upstream ITS IR frames";
  const auto outputName = result.kind == WorkflowKind::TrackerOnly ? (result.output == OutputPolicy::None ? "none" : "tracks") : result.output == OutputPolicy::All                  ? "tracks+clusters"
                                                                                                                               : result.output == OutputPolicy::TracksAndClusterROFs ? "tracks+cluster ROFs"
                                                                                                                               : result.output == OutputPolicy::ClusterROFs          ? "cluster ROFs"
                                                                                                                                                                                     : "none";
  LOGP(info, "MFT CA resolved: mode={} threads={} tracking={} input={} geometry={} IR source={} filter={} ROOT output={} MC={}",
       o2::itsmft::TrackingMode::toString(result.tracker.mode), result.tracker.nThreads,
       !result.runTracking ? "disabled" : result.tracker.mode == o2::itsmft::TrackingMode::Off ? "inactive"
                                                                                               : "active",
       inputName, result.tracker.geometry == GeometrySource::Full ? "full" : "MFT", irName, result.tracker.filterIRFrames, outputName, result.tracker.useMC);
  return result;
}
} // namespace o2::mft::ca
