// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file  TPCInterpolationSpec.cxx

#include <vector>

#include "DataFormatsITS/TrackITS.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "TPCInterpolationWorkflow/TPCInterpolationSpec.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{

void TPCInterpolationDPL::init(InitContext& ic)
{
  //-------- init geometry and field --------//
  o2::base::GeometryManager::loadGeometry();
  o2::base::Propagator::initFieldFromGRP("o2sim_grp.root");
  mTimer.Stop();
  mTimer.Reset();
  mInterpolation.init();
}

void TPCInterpolationDPL::run(ProcessingContext& pc)
{
  mTimer.Start(false);
  const auto tracksITS = pc.inputs().get<gsl::span<o2::its::TrackITS>>("trackITS");
  const auto tracksTPC = pc.inputs().get<gsl::span<o2::tpc::TrackTPC>>("trackTPC");
  const auto tracksITSTPC = pc.inputs().get<gsl::span<o2::dataformats::TrackTPCITS>>("match");
  const auto tracksTPCClRefs = pc.inputs().get<gsl::span<o2::tpc::TPCClRefElem>>("trackTPCClRefs");
  const auto trackMatchesTOF = pc.inputs().get<gsl::span<o2::dataformats::MatchInfoTOF>>("matchTOF"); // FIXME missing reader
  const auto clustersTOF = pc.inputs().get<gsl::span<o2::tof::Cluster>>("clustersTOF");

  const auto& inputsTPCclusters = getWorkflowTPCInput(pc);

  // pass input data to TrackInterpolation object
  mInterpolation.setITSTracksInp(tracksITS);
  mInterpolation.setTPCTracksInp(tracksTPC);
  mInterpolation.setTPCTrackClusIdxInp(tracksTPCClRefs);
  mInterpolation.setTPCClustersInp(&inputsTPCclusters->clusterIndex);
  mInterpolation.setTOFMatchesInp(trackMatchesTOF);
  mInterpolation.setITSTPCTrackMatchesInp(tracksITSTPC);
  mInterpolation.setTOFClustersInp(clustersTOF);

  if (mUseMC) {
    // possibly MC labels will be used to check filtering procedure performance before interpolation
    // not yet implemented
  }

  LOG(INFO) << "TPC Interpolation Workflow initialized. Start processing...";

  mInterpolation.process();

  pc.outputs().snapshot(Output{"GLO", "TPCINT_TRK", 0, Lifetime::Timeframe}, mInterpolation.getReferenceTracks());
  pc.outputs().snapshot(Output{"GLO", "TPCINT_RES", 0, Lifetime::Timeframe}, mInterpolation.getClusterResiduals());
  mTimer.Stop();
}

void TPCInterpolationDPL::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "TPC residuals extraction total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getTPCInterpolationSpec(bool useMC, const std::vector<int>& tpcClusLanes)
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;

  inputs.emplace_back("trackITS", "ITS", "TRACKS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackTPC", "TPC", "TRACKS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackTPCClRefs", "TPC", "CLUSREFS", 0, Lifetime::Timeframe);

  inputs.emplace_back("clusTPC", ConcreteDataTypeMatcher{"TPC", "CLUSTERNATIVE"}, Lifetime::Timeframe);

  inputs.emplace_back("match", "GLO", "TPCITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("matchTOF", "TOF", "MATCHINFOS", 0, Lifetime::Timeframe);
  inputs.emplace_back("clustersTOF", "TOF", "CLUSTERS", 0, Lifetime::Timeframe);

  if (useMC) {
    LOG(FATAL) << "MC usage must be disabled for this workflow, since it is not yet implemented";
    // are the MC inputs from ITS-TPC matching and TOF matching duplicates? if trackITSMCTR == matchTOFMCITS one of them should be removed
    inputs.emplace_back("trackITSMCTR", "GLO", "TPCITS_ITSMC", 0, Lifetime::Timeframe);
    inputs.emplace_back("trackTPCMCTR", "GLO", "TPCITS_TPCMC", 0, Lifetime::Timeframe);
    inputs.emplace_back("matchTOFMC", o2::header::gDataOriginTOF, "MATCHTOFINFOSMC", 0, Lifetime::Timeframe);
    inputs.emplace_back("matchTOFMCTPC", o2::header::gDataOriginTOF, "MATCHTPCINFOSMC", 0, Lifetime::Timeframe);
    inputs.emplace_back("matchTOFMCITS", o2::header::gDataOriginTOF, "MATCHITSINFOSMC", 0, Lifetime::Timeframe);
  }

  outputs.emplace_back("GLO", "TPCINT_TRK", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "TPCINT_RES", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "tpc-track-interpolation",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCInterpolationDPL>(useMC)},
    Options{}};
}

} // namespace tpc
} // namespace o2
