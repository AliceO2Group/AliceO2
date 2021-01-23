#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TFile.h"
#include "TTree.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsTOF/Cluster.h"
#include "TOFBase/Geo.h"
#endif

void findLabels(int itrk, int ientry)
{

  // macro to find the labels of a TPCITS track and the corresponding TOF cluster

  // getting the ITSTPCtracks
  TFile* fmatchITSTPC = new TFile("o2match_itstpc.root");
  TTree* matchTPCITS = (TTree*)fmatchITSTPC->Get("matchTPCITS");
  std::vector<o2::dataformats::TrackTPCITS>* mTracksArrayInp = new std::vector<o2::dataformats::TrackTPCITS>;
  matchTPCITS->SetBranchAddress("TPCITS", &mTracksArrayInp);
  matchTPCITS->GetEntry(ientry);

  // getting the TPC tracks
  TFile* ftracksTPC = new TFile("tpctracks.root");
  TTree* tpcTree = (TTree*)ftracksTPC->Get("tpcrec");
  std::vector<o2::tpc::TrackTPC>* mTPCTracksArrayInp = new std::vector<o2::tpc::TrackTPC>;
  tpcTree->SetBranchAddress("TPCTracks", &mTPCTracksArrayInp);
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mcTPC = new o2::dataformats::MCTruthContainer<o2::MCCompLabel>();
  tpcTree->SetBranchAddress("TPCTracksMCTruth", &mcTPC);

  // getting the ITS tracks
  TFile* ftracksITS = new TFile("o2trac_its.root");
  TTree* itsTree = (TTree*)ftracksITS->Get("o2sim");
  std::vector<o2::its::TrackITS>* mITSTracksArrayInp = new std::vector<o2::its::TrackITS>;
  itsTree->SetBranchAddress("ITSTrack", &mITSTracksArrayInp);
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mcITS = new o2::dataformats::MCTruthContainer<o2::MCCompLabel>();
  itsTree->SetBranchAddress("ITSTrackMCTruth", &mcITS);

  // getting the TOF clusters
  TFile* fclustersTOF = new TFile("tofclusters.root");
  TTree* tofClTree = (TTree*)fclustersTOF->Get("o2sim");
  std::vector<o2::tof::Cluster>* mTOFClustersArrayInp = new std::vector<o2::tof::Cluster>;
  tofClTree->SetBranchAddress("TOFCluster", &mTOFClustersArrayInp);
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mcTOF = new o2::dataformats::MCTruthContainer<o2::MCCompLabel>();
  tofClTree->SetBranchAddress("TOFClusterMCTruth", &mcTOF);

  tpcTree->GetEntry(0);
  tofClTree->GetEntry(0);

  o2::dataformats::TrackTPCITS trackITSTPC = mTracksArrayInp->at(itrk);
  int evIdxTPC = trackITSTPC.getRefTPC();
  Printf("matched TPCtrack  index = %d", evIdxTPC);
  int evIdxITS = trackITSTPC.getRefITS();
  Printf("matched ITStrack index = %d", evIdxITS);
  itsTree->GetEntry(evIdxITS);

  // getting the TPC labels
  const auto& labelsTPC = mcTPC->getLabels(evIdxTPC);
  for (int ilabel = 0; ilabel < labelsTPC.size(); ilabel++) {
    Printf("TPC label %d: trackID = %d, eventID = %d, sourceID = %d", ilabel, labelsTPC[ilabel].getTrackID(), labelsTPC[ilabel].getEventID(), labelsTPC[ilabel].getSourceID());
  }

  // getting the ITS labels
  const auto& labelsITS = mcITS->getLabels(evIdxITS);
  for (int ilabel = 0; ilabel < labelsITS.size(); ilabel++) {
    Printf("ITS label %d: trackID = %d, eventID = %d, sourceID = %d", ilabel, labelsITS[ilabel].getTrackID(), labelsITS[ilabel].getEventID(), labelsITS[ilabel].getSourceID());
  }

  // now checking if we have a corresponding cluster
  bool found = false;
  for (int tofClIndex = 0; tofClIndex < int(mTOFClustersArrayInp->size()); tofClIndex++) {
    o2::tof::Cluster tofCluster = mTOFClustersArrayInp->at(tofClIndex);
    const auto& labelsTOF = mcTOF->getLabels(tofClIndex);
    int trackIdTOF;
    int eventIdTOF;
    int sourceIdTOF;
    for (int ilabel = 0; ilabel < labelsTOF.size(); ilabel++) {
      if ((labelsTPC[0].getTrackID() == labelsTOF[ilabel].getTrackID() && labelsTPC[0].getEventID() == labelsTOF[ilabel].getEventID() && labelsTPC[0].getSourceID() == labelsTOF[ilabel].getSourceID()) || (labelsITS[0].getTrackID() == labelsTOF[ilabel].getTrackID() && labelsITS[0].getEventID() == labelsTOF[ilabel].getEventID() && labelsITS[0].getSourceID() == labelsTOF[ilabel].getSourceID())) {
        Printf("The corresponding TOF cluster is %d", tofClIndex);
        Printf("TOF label %d: trackID = %d, eventID = %d, sourceID = %d", ilabel, labelsTOF[ilabel].getTrackID(), labelsTOF[ilabel].getEventID(), labelsTOF[ilabel].getSourceID());
        found = true;
        int nContrChannels = tofCluster.getNumOfContributingChannels();
        int mainContrChannel = tofCluster.getMainContributingChannel();
        int* indices = new int[5];
        o2::tof::Geo::getVolumeIndices(mainContrChannel, indices);
        Printf("main contributing channel: sector = %d, plate = %d, strip = %d, padz = %d, padx = %d", indices[0], indices[1], indices[2], indices[3], indices[4]);
        break;
      }
    }
  }
  if (!found)
    Printf("No TOF cluster corresponding to this track was found");

  return;
}
