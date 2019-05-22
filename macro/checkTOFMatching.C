#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TFile.h"
#include "TTree.h"
#include "GlobalTracking/MatchTOF.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#endif

void checkTOFMatching()
{

  // macro to check the matching TOF-ITSTPC tracks

  // getting TOF info
  TFile* fmatchTOF = new TFile("o2match_tof.root");
  TTree* matchTOF = (TTree*)fmatchTOF->Get("matchTOF");
  std::vector<o2::dataformats::MatchInfoTOF>* TOFMatchInfo;
  TOFMatchInfo = new std::vector<o2::dataformats::MatchInfoTOF>;
  matchTOF->SetBranchAddress("TOFMatchInfo", &TOFMatchInfo);

  // getting the ITSTPCtracks
  TFile* fmatchITSTPC = new TFile("o2match_itstpc.root");
  TTree* matchTPCITS = (TTree*)fmatchITSTPC->Get("matchTPCITS");
  std::vector<o2::dataformats::TrackTPCITS>* mTracksArrayInp = new std::vector<o2::dataformats::TrackTPCITS>;
  matchTPCITS->SetBranchAddress("TPCITS", &mTracksArrayInp);

  // getting the TPC tracks
  TFile* ftracksTPC = new TFile("tpctracks.root");
  TTree* tpcTree = (TTree*)ftracksTPC->Get("events");
  std::vector<o2::tpc::TrackTPC>* mTPCTracksArrayInp = new std::vector<o2::tpc::TrackTPC>;
  tpcTree->SetBranchAddress("Tracks", &mTPCTracksArrayInp);
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mcTPC = new o2::dataformats::MCTruthContainer<o2::MCCompLabel>();
  tpcTree->SetBranchAddress("TracksMCTruth", &mcTPC);

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

  int nMatches = 0;
  int nGoodMatches = 0;
  int nBadMatches = 0;

  // now looping over the entries in the matching tree
  for (int ientry = 0; ientry < matchTOF->GetEntries(); ientry++) {
    matchTOF->GetEvent(ientry);
    matchTPCITS->GetEntry(ientry);
    // now looping over the matched tracks
    nMatches += TOFMatchInfo->size();
    for (int imatch = 0; imatch < TOFMatchInfo->size(); imatch++) {
      int indexITSTPCtrack = TOFMatchInfo->at(imatch).getTrackIndex();
      o2::dataformats::MatchInfoTOF infoTOF = TOFMatchInfo->at(imatch);
      int tofClIndex = infoTOF.getTOFClIndex();
      float chi2 = infoTOF.getChi2();
      Printf("\nentry in tree %d, matching %d, indexITSTPCtrack = %d, tofClIndex = %d, chi2 = %f", ientry, imatch, indexITSTPCtrack, tofClIndex, chi2);

      //      o2::MCCompLabel label = mcTOF->getElement(mcTOF->getMCTruthHeader(tofClIndex).index);
      const auto& labelsTOF = mcTOF->getLabels(tofClIndex);
      int trackIdTOF;
      int eventIdTOF;
      int sourceIdTOF;
      for (int ilabel = 0; ilabel < labelsTOF.size(); ilabel++) {
        Printf("TOF label %d: trackID = %d, eventID = %d, sourceID = %d", ilabel, labelsTOF[ilabel].getTrackID(), labelsTOF[ilabel].getEventID(), labelsTOF[ilabel].getSourceID());
        if (ilabel == 0) {
          trackIdTOF = labelsTOF[ilabel].getTrackID();
          eventIdTOF = labelsTOF[ilabel].getEventID();
          sourceIdTOF = labelsTOF[ilabel].getSourceID();
        }
      }
      o2::tof::Cluster tofCluster = mTOFClustersArrayInp->at(tofClIndex);
      int nContributingChannels = tofCluster.getNumOfContributingChannels();
      int mainContributingChannel = tofCluster.getMainContributingChannel();
      Printf("The TOF cluster has %d contributing channels, and the main one is %d", nContributingChannels, mainContributingChannel);
      int* indices = new int();
      o2::tof::Geo::getVolumeIndices(mainContributingChannel, indices);
      Printf("Indices of main contributing channel are %d, %d, %d, %d, %d", indices[0], indices[1], indices[2], indices[3], indices[4]);
      bool isUpLeft = tofCluster.isAdditionalChannelSet(o2::tof::Cluster::kUpLeft);
      bool isUp = tofCluster.isAdditionalChannelSet(o2::tof::Cluster::kUp);
      bool isUpRight = tofCluster.isAdditionalChannelSet(o2::tof::Cluster::kUpRight);
      bool isRight = tofCluster.isAdditionalChannelSet(o2::tof::Cluster::kRight);
      bool isDownRight = tofCluster.isAdditionalChannelSet(o2::tof::Cluster::kDownRight);
      bool isDown = tofCluster.isAdditionalChannelSet(o2::tof::Cluster::kDown);
      bool isDownLeft = tofCluster.isAdditionalChannelSet(o2::tof::Cluster::kDownLeft);
      bool isLeft = tofCluster.isAdditionalChannelSet(o2::tof::Cluster::kLeft);
      Printf("isUpLeft = %d, isUp = %d, isUpRight = %d, isRight = %d, isDownRight = %d, isDown = %d, isDownLeft = %d, isLeft = %d", isUpLeft, isUp, isUpRight, isRight, isDownRight, isDown, isDownLeft, isLeft);
      int* indexCont = new int();
      indexCont[0] = indices[0];
      indexCont[1] = indices[1];
      indexCont[2] = indices[2];
      indexCont[3] = indices[3];
      indexCont[4] = indices[4];
      int numberOfSecondaryContributingChannels = 0;
      int secondaryContributingChannel = -1;
      if (isDown) {
        indexCont[3]--;
        numberOfSecondaryContributingChannels++;
        secondaryContributingChannel = o2::tof::Geo::getIndex(indexCont);
        Printf("secondaryContributingChannel[down] = %d", secondaryContributingChannel);
        indexCont[3] = indices[3];
      }
      if (isDownRight) {
        indexCont[3]--;
        indexCont[4]++;
        numberOfSecondaryContributingChannels++;
        secondaryContributingChannel = o2::tof::Geo::getIndex(indexCont);
        Printf("secondaryContributingChannel[downright] = %d", secondaryContributingChannel);
        indexCont[3] = indices[3];
        indexCont[4] = indices[4];
      }
      if (isDownLeft) {
        indexCont[3]--;
        indexCont[4]--;
        numberOfSecondaryContributingChannels++;
        secondaryContributingChannel = o2::tof::Geo::getIndex(indexCont);
        Printf("secondaryContributingChannel[downleft] = %d", secondaryContributingChannel);
        indexCont[3] = indices[3];
        indexCont[4] = indices[4];
      }
      if (isUp) {
        indexCont[3]++;
        numberOfSecondaryContributingChannels++;
        secondaryContributingChannel = o2::tof::Geo::getIndex(indexCont);
        Printf("secondaryContributingChannel[up] = %d", secondaryContributingChannel);
        indexCont[3] = indices[3];
      }
      if (isUpRight) {
        indexCont[3]++;
        indexCont[4]++;
        numberOfSecondaryContributingChannels++;
        secondaryContributingChannel = o2::tof::Geo::getIndex(indexCont);
        Printf("secondaryContributingChannel[upright] = %d", secondaryContributingChannel);
        indexCont[3] = indices[3];
        indexCont[4] = indices[4];
      }
      if (isUpLeft) { // increase padZ
        indexCont[3]++;
        indexCont[4]--;
        numberOfSecondaryContributingChannels++;
        secondaryContributingChannel = o2::tof::Geo::getIndex(indexCont);
        Printf("secondaryContributingChannel[upleft] = %d", secondaryContributingChannel);
        indexCont[3] = indices[3];
        indexCont[4] = indices[4];
      }
      if (isRight) { // increase padX
        indexCont[4]++;
        numberOfSecondaryContributingChannels++;
        secondaryContributingChannel = o2::tof::Geo::getIndex(indexCont);
        Printf("secondaryContributingChannel[right] = %d", secondaryContributingChannel);
        indexCont[4] = indices[4];
      }
      if (isLeft) { // decrease padX
        indexCont[4]--;
        numberOfSecondaryContributingChannels++;
        secondaryContributingChannel = o2::tof::Geo::getIndex(indexCont);
        Printf("secondaryContributingChannel[left] = %d", secondaryContributingChannel);
        indexCont[4] = indices[4];
      }
      Printf("Total number of secondary channels= %d", numberOfSecondaryContributingChannels);

      o2::dataformats::TrackTPCITS trackITSTPC = mTracksArrayInp->at(indexITSTPCtrack);
      const o2::dataformats::EvIndex<int, int>& evIdxTPC = trackITSTPC.getRefTPC();
      Printf("matched TPCtrack: eventID = %d, indexID = %d", evIdxTPC.getEvent(), evIdxTPC.getIndex());
      const o2::dataformats::EvIndex<int, int>& evIdxITS = trackITSTPC.getRefITS();
      Printf("matched ITStrack: eventID = %d, indexID = %d", evIdxITS.getEvent(), evIdxITS.getIndex());
      itsTree->GetEntry(evIdxITS.getEvent());

      // getting the TPC labels
      const auto& labelsTPC = mcTPC->getLabels(evIdxTPC.getIndex());
      for (int ilabel = 0; ilabel < labelsTPC.size(); ilabel++) {
        Printf("TPC label %d: trackID = %d, eventID = %d, sourceID = %d", ilabel, labelsTPC[ilabel].getTrackID(), labelsTPC[ilabel].getEventID(), labelsTPC[ilabel].getSourceID());
      }

      // getting the ITS labels
      const auto& labelsITS = mcITS->getLabels(evIdxITS.getIndex());
      for (int ilabel = 0; ilabel < labelsITS.size(); ilabel++) {
        Printf("ITS label %d: trackID = %d, eventID = %d, sourceID = %d", ilabel, labelsITS[ilabel].getTrackID(), labelsITS[ilabel].getEventID(), labelsITS[ilabel].getSourceID());
      }

      bool bMatched = kFALSE;
      for (int ilabel = 0; ilabel < labelsTOF.size(); ilabel++) {
        if ((abs(labelsTPC[0].getTrackID()) == labelsTOF[ilabel].getTrackID() && labelsTPC[0].getEventID() == labelsTOF[ilabel].getEventID() && labelsTPC[0].getSourceID() == labelsTOF[ilabel].getSourceID()) || (labelsITS[0].getTrackID() == labelsTOF[ilabel].getTrackID() && labelsITS[0].getEventID() == labelsTOF[ilabel].getEventID() && labelsITS[0].getSourceID() == labelsTOF[ilabel].getSourceID())) {
          nGoodMatches++;
          bMatched = kTRUE;
          break;
        }
      }
      if (!bMatched)
        nBadMatches++;

      bool TPCfound = false;
      bool ITSfound = false;
      for (int i = 0; i < mTracksArrayInp->size(); i++) {
        o2::dataformats::TrackTPCITS trackITSTPC = mTracksArrayInp->at(i);
        const o2::dataformats::EvIndex<int, int>& evIdxTPCcheck = trackITSTPC.getRefTPC();
        const o2::dataformats::EvIndex<int, int>& evIdxITScheck = trackITSTPC.getRefITS();
        itsTree->GetEntry(evIdxITScheck.getEvent());
        const auto& labelsTPCcheck = mcTPC->getLabels(evIdxTPCcheck.getIndex());
        for (int ilabel = 0; ilabel < labelsTPCcheck.size(); ilabel++) {
          if (abs(labelsTPCcheck[ilabel].getTrackID()) == trackIdTOF && labelsTPCcheck[ilabel].getEventID() == eventIdTOF && labelsTPCcheck[ilabel].getSourceID() == sourceIdTOF) {
            Printf("The TPC track that should have been matched to TOF is number %d", i);
            TPCfound = true;
          }
        }
        const auto& labelsITScheck = mcITS->getLabels(evIdxITScheck.getIndex());
        for (int ilabel = 0; ilabel < labelsITScheck.size(); ilabel++) {
          if (labelsITScheck[ilabel].getTrackID() == trackIdTOF && labelsITScheck[ilabel].getEventID() == eventIdTOF && labelsITScheck[ilabel].getSourceID() == sourceIdTOF) {
            Printf("The ITS track that should have been matched to TOF is number %d", i);
            ITSfound = true;
          }
        }
      }
      if (!TPCfound)
        Printf("There is no TPC track found that should have corresponded to this TOF cluster!");
      if (!ITSfound)
        Printf("There is no ITS track found that should have corresponded to this TOF cluster!");
    }
  }

  Printf("Number of      matches = %d", nMatches);
  Printf("Number of GOOD matches = %d (%.2f)", nGoodMatches, (float)nGoodMatches / nMatches);
  Printf("Number of BAD  matches = %d (%.2f)", nBadMatches, (float)nBadMatches / nMatches);

  return;
}
