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
#include "Headers/DataHeader.h"
#include "DataFormatsTOF/Cluster.h"
#include "GlobalTracking/MatchTOF.h"

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

 public:
  void init(framework::InitContext& ic)
  {
    // nothing special to be set up
  }

  void run(framework::ProcessingContext& pc)
  {
    static bool finished = false;
    if (finished) {
      return;
    }

  //>>>---------- attach input data --------------->>>
  auto tracks = pc.inputs().get<std::vector<o2::dataformats::TrackTPCITS>*>("trackITSTPC");
  auto clusters = pc.inputs().get<std::vector<o2::tof::Cluster>*>("tofclusters");

  //-------- init geometry and field --------//
  std::string path = "./";
  std::string inputGeom = "O2geometry.root";
  std::string inputGRP = "o2sim_grp.root";

  //  o2::base::GeometryManager::loadGeometry(path + inputGeom, "FAIRGeom");
  //  o2::base::Propagator::initFieldFromGRP(path + inputGRP);

  // call actual matching info routine
#ifdef _ALLOW_DEBUG_TREES_
  mMatcher.setDebugTreeFileName(path + mMatcher.getDebugTreeFileName());
  mMatcher.setDebugFlag(o2::globaltracking::MatchTOF::MatchTreeAll);
#endif
  std::vector<o2::dataformats::TrackTPCITS> tracksRO;
  for (int i = 0; i < tracks->size(); i++) {
    tracksRO.emplace_back(tracks->at(i));
  }
  std::vector<o2::tof::Cluster> clustersRO;
   for (int i = 0; i < clusters->size(); i++) {
     clustersRO.emplace_back(clusters->at(i));
  } 

  mMatcher.initWorkflow(&tracksRO , &clustersRO);

  mMatcher.run();

  // in run_match_tof aggiugnere esplicitamente la chiamata a fill del tree (nella classe MatchTOF) e il metodo per leggere i vettori di output

  //...
    // LOG(INFO) << "TOF CLUSTERER : TRANSFORMED " << digits->size()
    //           << " DIGITS TO " << mClustersArray.size() << " CLUSTERS";


    // send matching-info
    pc.outputs().snapshot(Output{ "TOF", "MATCHINFOS", 0, Lifetime::Timeframe }, mMatchedTracks);
    pc.outputs().snapshot(Output{ "TOF", "CALIBINFOS", 0, Lifetime::Timeframe }, mCalibInfoTOF);

    // declare done
    finished = true;
    pc.services().get<ControlService>().readyToQuit(false);
  }

 private:
  o2::globaltracking::MatchTOF mMatcher;    ///< Cluster finder

  std::vector<std::pair<evIdx, o2::dataformats::MatchInfoTOF>> mMatchedTracks; ///< this is the output of the matching
  std::vector<o2::dataformats::CalibInfoTOF> mCalibInfoTOF; ///< Array of calib-info
  // std::vector<o2::dataformats::MatchInfoTOF> mMatchedTracks; ///< this is the output of the matching
  // std::vector<o2::dataformats::CalibInfoTOF> mCalibInfoTOF; ///< Array of calib-info
};

o2::framework::DataProcessorSpec getTOFRecoWorkflowSpec()
{
    // return DataProcessorSpec{
    //   "TOFRecoWorkflow",
    //     Inputs{ InputSpec{ "trackITSTPC", "ITSTPC", "TRACKS", 0, Lifetime::Timeframe } , InputSpec{ "tofclusters", "TOF", "CLUSTERS", 0, Lifetime::Timeframe }},
    //   Outputs{ OutputSpec{ "TOF", "MATCHINFOS", 0, Lifetime::Timeframe },
    //            OutputSpec{ "TOF", "CALIBINFOS", 0, Lifetime::Timeframe } },
    //   AlgorithmSpec{ adaptFromTask<TOFDPLRecoWorkflowTask>() },
    //   Options{ /* for the moment no options */ }
    // };
}

} // end namespace tof
} // end namespace o2
