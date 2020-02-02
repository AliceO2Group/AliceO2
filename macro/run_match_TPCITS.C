#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TFile.h>
#include <TChain.h>
#include <TTree.h>
#include <TGeoGlobalMagField.h>
#include <string>
#include <FairLogger.h>

#include "Field/MagneticField.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"

#include "GlobalTracking/MatchTPCITS.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#endif

void run_match_TPCITS(std::string path = "./", std::string outputfile = "o2match_itstpc.root",
                      std::string inputTracksITS = "o2trac_its.root",
                      std::string inputTracksTPC = "tpctracks.root",
                      std::string inputClustersITS = "o2clus_its.root",
                      std::string inputClustersTPC = "tpc-native-clusters.root",
                      std::string inputFITInfo = "o2reco_ft0.root", // optional FIT (T0) info
                      std::string inputGeom = "O2geometry.root",
                      std::string inputGRP = "o2sim_grp.root")
{

  o2::globaltracking::MatchTPCITS matching;

  if (path.back() != '/') {
    path += '/';
  }

  //>>>---------- attach input data --------------->>>
  TChain itsTracks("o2sim");
  itsTracks.AddFile((path + inputTracksITS).c_str());
  matching.setInputTreeITSTracks(&itsTracks);

  TChain tpcTracks("events");
  tpcTracks.AddFile((path + inputTracksTPC).c_str());
  matching.setInputTreeTPCTracks(&tpcTracks);

  TChain itsClusters("o2sim");
  itsClusters.AddFile((path + inputClustersITS).c_str());
  matching.setInputTreeITSClusters(&itsClusters);

  bool canUseFIT = false;
  TChain fitInfo("o2sim");
  if (!inputFITInfo.empty()) {
    if (!gSystem->AccessPathName((path + inputFITInfo).c_str())) {
      fitInfo.AddFile((path + inputFITInfo).c_str());
      matching.setInputTreeFITInfo(&fitInfo);
      canUseFIT = true;
    } else {
      LOG(ERROR) << "ATTENTION: FIT input " << inputFITInfo << " requested but not available";
    }
  }

  o2::tpc::ClusterNativeHelper::Reader tcpClusterReader;
  tcpClusterReader.init(inputClustersTPC.c_str());
  matching.setInputTPCClustersReader(&tcpClusterReader);
  //<<<---------- attach input data ---------------<<<

  // create/attach output tree
  TFile outFile((path + outputfile).c_str(), "recreate");
  TTree outTree("matchTPCITS", "Matched TPC-ITS tracks");
  matching.setOutputTree(&outTree);

#ifdef _ALLOW_DEBUG_TREES_
  matching.setDebugTreeFileName(path + matching.getDebugTreeFileName());
  // dump accepted pairs only
  matching.setDebugFlag(o2::globaltracking::MatchTPCITS::MatchTreeAccOnly);
  // dump all checked pairs
  // matching.setDebugFlag(o2::globaltracking::MatchTPCITS::MatchTreeAll);
  // dump winner matches
  matching.setDebugFlag(o2::globaltracking::MatchTPCITS::WinnerMatchesTree);
#endif

  //-------- init geometry and field --------//
  o2::base::GeometryManager::loadGeometry(path + inputGeom, "FAIRGeom");
  o2::base::Propagator::initFieldFromGRP(path + inputGRP);

  //-------------------- settings -----------//
  const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
  matching.setITSROFrameLengthMUS(alpParams.roFrameLength / 1.e3); // ITS ROFrame duration in \mus
  // Note: parameters are set via o2::globaltracking::MatchITSTPCParams
  matching.init();

  matching.run();

  outFile.cd();
  outTree.Write();
  outFile.Close();
}
