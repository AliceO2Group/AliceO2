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
                      std::string inputGRP = "o2sim_grp.root")
{

  o2::globaltracking::MatchTPCITS matching;

  if (path.back() != '/') {
    path += '/';
  }

  //>>>---------- attach input data --------------->>>
  o2::tpc::ClusterNativeHelper::Reader* mTPCClusterReader = nullptr;     ///< TPC cluster reader
  std::unique_ptr<o2::tpc::ClusterNativeAccess> mTPCClusterIdxStructOwn; ///< used in case of tree-based IO
  std::unique_ptr<o2::tpc::ClusterNative[]> mTPCClusterBufferOwn;        ///< buffer for clusters in mTPCClusterIdxStructOwn
  o2::tpc::MCLabelContainer mTPCClusterMCBufferOwn;                      ///< buffer for mc labels

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
  o2::base::GeometryManager::loadGeometry(path);
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

void setupTF(int entry)
{
  std::string mITSTrackBranchName = "ITSTrack";                ///< name of branch containing input ITS tracks
  std::string mITSTrackClusIdxBranchName = "ITSTrackClusIdx";  ///< name of branch containing input ITS tracks cluster indices
  std::string mITSTrackROFRecBranchName = "ITSTracksROF";      ///< name of branch containing input ITS tracks ROFRecords
  std::string mTPCTrackBranchName = "Tracks";                  ///< name of branch containing input TPC tracks
  std::string mTPCTrackClusIdxBranchName = "ClusRefs";         ///< name of branch containing input TPC tracks cluster references
  std::string mITSClusterBranchName = "ITSCluster";            ///< name of branch containing input ITS clusters
  std::string mITSClusMCTruthBranchName = "ITSClusterMCTruth"; ///< name of branch containing input ITS clusters MC
  std::string mITSClusterROFRecBranchName = "ITSClustersROF";  ///< name of branch containing input ITS clusters ROFRecords
  std::string mITSMCTruthBranchName = "ITSTrackMCTruth";       ///< name of branch containing ITS MC labels
  std::string mTPCMCTruthBranchName = "TracksMCTruth";         ///< name of branch containing input TPC tracks
  std::string mFITInfoBranchName = "FT0Cluster";               ///< name of branch containing input FIT Info

  if (!mDPLIO) { // init
    attachInputTrees();

    TTree* mOutputTree = nullptr; ///< output tree for matched tracks

    // create output branch
    if (mOutputTree) {
      mOutputTree->Branch(NAMES::TPCITS_TracksBranchName.data(), &mMatchedTracks);
      if (mMCTruthON) {
        mOutputTree->Branch(NAMES::TPCITS_ITSMCTruthBranchName.data(), &mOutITSLabels);
        mOutputTree->Branch(NAMES::TPCITS_TPCMCTruthBranchName.data(), &mOutTPCLabels);
      }
    } else {
      LOG(ERROR) << "Output tree is not attached, matched tracks will not be stored";
    }
  }
  std::vector<o2::itsmft::Cluster> mITSClustersBuffer; ///< input ITS clusters buffer for tree IO
  std::vector<o2::itsmft::ROFRecord> mITSClusterROFRecBuffer;
  MCLabCont mITSClsLabelsBuffer;

  std::vector<o2::tpc::TrackTPC>* mTPCTracksArrayPtr = nullptr;             ///< input TPC tracks from tree
  std::vector<o2::tpc::TPCClRefElem>* mTPCTrackClusIdxPtr = nullptr;        ///< input TPC track cluster indices from tree
  std::vector<o2::itsmft::ROFRecord>* mITSTrackROFRecPtr = nullptr;         ///< input ITS tracks ROFRecord from tree
  std::vector<o2::its::TrackITS>* mITSTracksArrayPtr = nullptr;             ///< input ITS tracks read from tree
  std::vector<int>* mITSTrackClusIdxPtr = nullptr;                          ///< input ITS track cluster indices from tree
  const std::vector<o2::itsmft::Cluster>* mITSClustersArrayPtr = nullptr;   ///< input ITS clusters from tree
  const std::vector<o2::itsmft::ROFRecord>* mITSClusterROFRecPtr = nullptr; ///< input ITS clusters ROFRecord from tree
  const std::vector<o2::ft0::RecPoints>* mFITInfoPtr = nullptr;             ///< optional input FIT info from the tree

  mTimerIO.Start(false);

  mTreeITSTracks->GetEntry(0);
  mITSTrackROFRec = gsl::span<const o2::itsmft::ROFRecord>(mITSTrackROFRecPtr->data(), mITSTrackROFRecPtr->size());
  mTreeITSClusters->GetEntry(0);
  mITSClustersArray = gsl::span<const o2::itsmft::Cluster>(mITSClustersArrayPtr->data(), mITSClustersArrayPtr->size());
  mITSClusterROFRec = gsl::span<const o2::itsmft::ROFRecord>(mITSClusterROFRecPtr->data(), mITSClusterROFRecPtr->size());
  mITSTracksArray = gsl::span<const o2::its::TrackITS>(mITSTracksArrayPtr->data(), mITSTracksArrayPtr->size());
  mITSTrackClusIdx = gsl::span<const int>(mITSTrackClusIdxPtr->data(), mITSTrackClusIdxPtr->size());

  mTreeTPCTracks->GetEntry(0);
  LOG(DEBUG) << "Loading TPC tracks: " << mTPCTracksArrayPtr->size() << " tracks";
  mTPCTracksArray = gsl::span<const o2::tpc::TrackTPC>(mTPCTracksArrayPtr->data(), mTPCTracksArrayPtr->size());
  mTPCTrackClusIdx = gsl::span<const o2::tpc::TPCClRefElem>(mTPCTrackClusIdxPtr->data(), mTPCTrackClusIdxPtr->size());

  mTPCClusterReader->read(0);
  mTPCClusterReader->fillIndex(*mTPCClusterIdxStructOwn.get(), mTPCClusterBufferOwn, mTPCClusterMCBufferOwn);
  mTPCClusterIdxStruct = mTPCClusterIdxStructOwn.get();

  mTreeFITInfo->GetEntry(0);
  mFITInfo = gsl::span<const o2::ft0::RecPoints>(mFITInfoPtr->data(), mFITInfoPtr->size()); // FT0 not yet POD

  mTimerIO.Stop();

  if (mMatchedTracks.size() && mOutputTree) {
    mTimerIO.Start(false);
    mOutputTree->Fill();
    mTimerIO.Stop();
  }
}
