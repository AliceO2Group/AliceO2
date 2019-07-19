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

 public:
  explicit TOFDPLRecoWorkflowTask(bool useMC) : mUseMC(useMC) {}

  void init(framework::InitContext& ic)
  {
    // nothing special to be set up
    o2::base::GeometryManager::loadGeometry("./O2geometry.root", "FAIRGeom");
    o2::base::Propagator::initFieldFromGRP("o2sim_grp.root");
  }

  void run(framework::ProcessingContext& pc)
  {
    static bool finished = false;
    if (finished) {
      return;
    }

    //>>>---------- attach input data --------------->>>
    auto tracks = pc.inputs().get<std::vector<o2::dataformats::TrackTPCITS>*>("globaltrack");
    auto clusters = pc.inputs().get<std::vector<o2::tof::Cluster>*>("tofcluster");

    o2::dataformats::MCTruthContainer<o2::MCCompLabel> toflab;
    auto itslab = std::make_shared<std::vector<o2::MCCompLabel>>();
    auto tpclab = std::make_shared<std::vector<o2::MCCompLabel>>();

    if (mUseMC) {
      auto toflabel = pc.inputs().get<o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("tofclusterlabel");
      auto itslabel = pc.inputs().get<std::vector<o2::MCCompLabel>*>("itstracklabel");
      auto tpclabel = pc.inputs().get<std::vector<o2::MCCompLabel>*>("tpctracklabel");
      toflab = std::move(*toflabel);
      *itslab.get() = std::move(*itslabel);
      *tpclab.get() = std::move(*tpclabel);
    }

    //-------- init geometry and field --------//
    // std::string path = "./";
    // std::string inputGeom = "O2geometry.root";
    // std::string inputGRP = "o2sim_grp.root";

    //  o2::base::GeometryManager::loadGeometry(path + inputGeom, "FAIRGeom");
    //  o2::base::Propagator::initFieldFromGRP(path + inputGRP);

    // call actual matching info routine
    //#ifdef _ALLOW_DEBUG_TREES_
    //  mMatcher.setDebugTreeFileName(path + mMatcher.getDebugTreeFileName());
    //  mMatcher.setDebugFlag(o2::globaltracking::MatchTOF::MatchTreeAll);
    //#endif

    // we do a copy of the input but we are looking for a way to avoid it (current problem in conversion form unique_ptr to *)

    auto tracksRO = std::make_shared<std::vector<o2::dataformats::TrackTPCITS>>();
    //  std::vector<o2::dataformats::TrackTPCITS> tracksRO;
    *tracksRO.get() = std::move(*tracks);
    // for (int i = 0; i < tracks->size(); i++) {
    //   tracksRO.emplace_back(tracks->at(i));
    // }
    auto clustersRO = std::make_shared<std::vector<o2::tof::Cluster>>();
    //std::vector<o2::tof::Cluster> clustersRO;
    *clustersRO.get() = std::move(*clusters);
    //  for (int i = 0; i < clusters->size(); i++) {
    //    clustersRO.emplace_back(clusters->at(i));
    // }

    if (mUseMC)
      mMatcher.initWorkflow(tracksRO.get(), clustersRO.get(), &toflab, itslab.get(), tpclab.get());
    else
      mMatcher.initWorkflow(tracksRO.get(), clustersRO.get(), nullptr, nullptr, nullptr);

    mMatcher.run();

    // in run_match_tof aggiugnere esplicitamente la chiamata a fill del tree (nella classe MatchTOF) e il metodo per leggere i vettori di output

    //...
    // LOG(INFO) << "TOF CLUSTERER : TRANSFORMED " << digits->size()
    //           << " DIGITS TO " << mClustersArray.size() << " CLUSTERS";

    // send matching-info
    pc.outputs().snapshot(Output{ "TOF", "MATCHINFOS", 0, Lifetime::Timeframe }, mMatcher.getMatchedTrackVector());
    if (mUseMC) {
      pc.outputs().snapshot(Output{ "TOF", "MATCHTOFINFOSMC", 0, Lifetime::Timeframe }, mMatcher.getMatchedTOFLabelsVector());
      pc.outputs().snapshot(Output{ "TOF", "MATCHTPCINFOSMC", 0, Lifetime::Timeframe }, mMatcher.getMatchedTPCLabelsVector());
      pc.outputs().snapshot(Output{ "TOF", "MATCHITSINFOSMC", 0, Lifetime::Timeframe }, mMatcher.getMatchedITSLabelsVector());
    }
    pc.outputs().snapshot(Output{ "TOF", "CALIBINFOS", 0, Lifetime::Timeframe }, mMatcher.getCalibVector());

    // declare done
    finished = true;
    pc.services().get<ControlService>().readyToQuit(false);
  }

 private:
  o2::globaltracking::MatchTOF mMatcher; ///< Cluster finder
};

o2::framework::DataProcessorSpec getTOFRecoWorkflowSpec(bool useMC)
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;
  inputs.emplace_back("tofcluster", "TOF", "CLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("globaltrack", "GLO", "TPCITS", 0, Lifetime::Timeframe);
  if (useMC) {
    inputs.emplace_back("tofclusterlabel", "TOF", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("itstracklabel", "GLO", "TPCITS_ITSMC", 0, Lifetime::Timeframe);
    inputs.emplace_back("tpctracklabel", "GLO", "TPCITS_TPCMC", 0, Lifetime::Timeframe);
  }

  outputs.emplace_back("TOF", "MATCHINFOS", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back("TOF", "MATCHTOFINFOSMC", 0, Lifetime::Timeframe);
    outputs.emplace_back("TOF", "MATCHTPCINFOSMC", 0, Lifetime::Timeframe);
    outputs.emplace_back("TOF", "MATCHITSINFOSMC", 0, Lifetime::Timeframe);
  }
  outputs.emplace_back("TOF", "CALIBINFOS", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "TOFRecoWorkflow",
    inputs,
    outputs,
    AlgorithmSpec{ adaptFromTask<TOFDPLRecoWorkflowTask>(useMC) },
    Options{ /* for the moment no options */ }
  };
}

} // end namespace tof
} // end namespace o2
