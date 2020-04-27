// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TOFWorkflow/RecoWorkflowSpec.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Framework/Task.h"
#include "Framework/SerializationMethods.h"
#include "Headers/DataHeader.h"
#include "DataFormatsTOF/Cluster.h"
#include "GlobalTracking/MatchTOF.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include <gsl/span>

// from FIT
#include "DataFormatsFT0/RecPoints.h"

#include <memory> // for make_shared, make_unique, unique_ptr
#include <vector>

using namespace o2::framework;

namespace o2
{
namespace tof
{

// use the tasking system of DPL
// just need to implement 2 special methods init + run (there is no need to inherit from anything)
class TOFDPLRecoWorkflowTask
{
  using evIdx = o2::dataformats::EvIndex<int, int>;
  using MatchOutputType = std::vector<o2::dataformats::MatchInfoTOF>;

  bool mUseMC = true;
  bool mUseFIT = false;

 public:
  explicit TOFDPLRecoWorkflowTask(bool useMC, bool useFIT) : mUseMC(useMC), mUseFIT(useFIT) {}

  void init(framework::InitContext& ic)
  {
    // nothing special to be set up
    o2::base::GeometryManager::loadGeometry();
    o2::base::Propagator::initFieldFromGRP("o2sim_grp.root");
  }

  void run(framework::ProcessingContext& pc)
  {
    static bool finished = false;
    if (finished) {
      return;
    }

    //>>>---------- attach input data --------------->>>
    const auto clustersRO = pc.inputs().get<gsl::span<o2::tof::Cluster>>("tofcluster");
    const auto tracksRO = pc.inputs().get<gsl::span<o2::dataformats::TrackTPCITS>>("globaltrack");

    if (mUseFIT) {
      // Note: the particular variable will go out of scope, but the span is passed by copy to the
      // worker and the underlying memory is valid throughout the whole computation
      auto recPoints = std::move(pc.inputs().get<gsl::span<o2::ft0::RecPoints>>("fitrecpoints"));
      mMatcher.setFITRecPoints(recPoints);
      LOG(INFO) << "TOF Reco Workflow pulled " << recPoints.size() << " FIT RecPoints";
    }

    //-------- init geometry and field --------//
    // std::string path = "./";
    // std::string inputGeom = "O2geometry.root";
    // std::string inputGRP = "o2sim_grp.root";

    //  o2::base::GeometryManager::loadGeometry(path);
    //  o2::base::Propagator::initFieldFromGRP(path + inputGRP);

    // call actual matching info routine
    //#ifdef _ALLOW_DEBUG_TREES_
    //  mMatcher.setDebugTreeFileName(path + mMatcher.getDebugTreeFileName());
    //  mMatcher.setDebugFlag(o2::globaltracking::MatchTOF::MatchTreeAll);
    //#endif

    // we do a copy of the input but we are looking for a way to avoid it (current problem in conversion form unique_ptr to *)

    gsl::span<const o2::MCCompLabel> itslab;
    gsl::span<const o2::MCCompLabel> tpclab;
    o2::dataformats::MCTruthContainer<o2::MCCompLabel> toflab;
    if (mUseMC) {
      const auto toflabel = pc.inputs().get<o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("tofclusterlabel");
      itslab = pc.inputs().get<gsl::span<o2::MCCompLabel>>("itstracklabel");
      tpclab = pc.inputs().get<gsl::span<o2::MCCompLabel>>("tpctracklabel");
      toflab = std::move(*toflabel);

      mMatcher.initWorkflow(&tracksRO, &clustersRO, &toflab, &itslab, &tpclab);
    } else {
      mMatcher.initWorkflow(&tracksRO, &clustersRO, nullptr, nullptr, nullptr);
    }
    mMatcher.run();

    // in run_match_tof aggiugnere esplicitamente la chiamata a fill del tree (nella classe MatchTOF) e il metodo per leggere i vettori di output

    //...
    // LOG(INFO) << "TOF CLUSTERER : TRANSFORMED " << digits->size()
    //           << " DIGITS TO " << mClustersArray.size() << " CLUSTERS";

    // send matching-info
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MATCHINFOS", 0, Lifetime::Timeframe}, mMatcher.getMatchedTrackVector());
    if (mUseMC) {
      pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MATCHTOFINFOSMC", 0, Lifetime::Timeframe}, mMatcher.getMatchedTOFLabelsVector());
      pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MATCHTPCINFOSMC", 0, Lifetime::Timeframe}, mMatcher.getMatchedTPCLabelsVector());
      pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MATCHITSINFOSMC", 0, Lifetime::Timeframe}, mMatcher.getMatchedITSLabelsVector());
    }
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "CALIBINFOS", 0, Lifetime::Timeframe}, mMatcher.getCalibVector());

    // declare done
    finished = true;
    //pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    pc.services().get<ControlService>().endOfStream();
  }

 private:
  o2::globaltracking::MatchTOF mMatcher; ///< Cluster finder
};

o2::framework::DataProcessorSpec getTOFRecoWorkflowSpec(bool useMC, bool useFIT)
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;
  inputs.emplace_back("tofcluster", o2::header::gDataOriginTOF, "CLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("globaltrack", "GLO", "TPCITS", 0, Lifetime::Timeframe);
  if (useMC) {
    inputs.emplace_back("tofclusterlabel", o2::header::gDataOriginTOF, "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("itstracklabel", "GLO", "TPCITS_ITSMC", 0, Lifetime::Timeframe);
    inputs.emplace_back("tpctracklabel", "GLO", "TPCITS_TPCMC", 0, Lifetime::Timeframe);
  }

  if (useFIT) {
    inputs.emplace_back("fitrecpoints", o2::header::gDataOriginFT0, "RECPOINTS", 0, Lifetime::Timeframe);
  }

  outputs.emplace_back(o2::header::gDataOriginTOF, "MATCHINFOS", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back(o2::header::gDataOriginTOF, "MATCHTOFINFOSMC", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTOF, "MATCHTPCINFOSMC", 0, Lifetime::Timeframe);
    outputs.emplace_back(o2::header::gDataOriginTOF, "MATCHITSINFOSMC", 0, Lifetime::Timeframe);
  }
  outputs.emplace_back(o2::header::gDataOriginTOF, "CALIBINFOS", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "TOFRecoWorkflow",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TOFDPLRecoWorkflowTask>(useMC, useFIT)},
    Options{/* for the moment no options */}};
}

} // end namespace tof
} // end namespace o2
