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
  std::vector<std::pair<int, o2::dataformats::MatchInfoTOF>>* TOFMatchInfo;
  TOFMatchInfo = new std::vector<std::pair<int, o2::dataformats::MatchInfoTOF>>;
  matchTOF->SetBranchAddress("TOFMatchInfo", &TOFMatchInfo);

  // getting the ITSTPCtracks
  TFile* fmatchITSTPC = new TFile("o2match_itstpc.root");
  TTree* matchTPCITS = (TTree*)fmatchITSTPC->Get("matchTPCITS");
  std::vector<o2::dataformats::TrackTPCITS>* mTracksArrayInp = new std::vector<o2::dataformats::TrackTPCITS>;
  matchTPCITS->SetBranchAddress("TPCITS", &mTracksArrayInp);

  // getting the TPC tracks
  TFile* ftracksTPC = new TFile("tpctracks.root");
  TTree* tpcTree = (TTree*)ftracksTPC->Get("events");
  std::vector<o2::TPC::TrackTPC>* mTPCTracksArrayInp = new std::vector<o2::TPC::TrackTPC>;
  tpcTree->SetBranchAddress("TPCTracks", &mTPCTracksArrayInp);
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mcTPC = new o2::dataformats::MCTruthContainer<o2::MCCompLabel>();
  tpcTree->SetBranchAddress("TPCTracksMCTruth", &mcTPC);

  // getting the ITS tracks
  TFile* ftracksITS = new TFile("o2trac_its.root");
  TTree* itsTree = (TTree*)ftracksITS->Get("o2sim");
  std::vector<o2::ITS::TrackITS>* mITSTracksArrayInp = new std::vector<o2::ITS::TrackITS>;
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
      int indexITSTPCtrack = TOFMatchInfo->at(imatch).first;
      o2::dataformats::MatchInfoTOF infoTOF = TOFMatchInfo->at(imatch).second;
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
      int* indices;
      o2::tof::Geo::getVolumeIndices(mainContributingChannel, indices);
      Printf("Indices of main contributing channel are %d, %d, %d, %d, %d", indices[0], indices[1], indices[2], indices[3], indices[4]);
      int* secondaryContributingChannels = new int[nContributingChannels];
      for (int ich = 1; ich < nContributingChannels; ich++) {
        bool isUpLeft = tofCluster.isUpLeftContributing();
        bool isUp = tofCluster.isUpContributing();
        bool isUpRight = tofCluster.isUpRightContributing();
        bool isRight = tofCluster.isRightContributing();
        bool isDownRight = tofCluster.isDownRightContributing();
        bool isDown = tofCluster.isDownContributing();
        bool isDownLeft = tofCluster.isDownLeftContributing();
        bool isLeft = tofCluster.isLeftContributing();
        Printf("isUpLeft = %d, isUp = %d, isUpRight = %d, isRight = %d, isDownRight = %d, isDown = %d, isDownLeft = %d, isLeft = %d", isUpLeft, isUp, isUpRight, isRight, isDownRight, isDown, isDownLeft, isLeft);
        int* indexCont = new int();
        indexCont[0] = indices[0];
        indexCont[1] = indices[1];
        indexCont[2] = indices[2];
        indexCont[3] = indices[3];
        indexCont[4] = indices[4];
        if (isDown || isDownRight || isDownLeft) { // decrease padZ
          indexCont[3]--;
        }
        if (isUp || isUpRight || isUpLeft) { // decrease padZ
          indexCont[3]++;
        }
        if (isRight || isDownRight || isUpRight) { // decrease padZ
          indexCont[4]++;
        }
        if (isLeft || isDownLeft || isUpLeft) { // decrease padZ
          indexCont[4]--;
        }
        secondaryContributingChannels[ich - 1] = o2::tof::Geo::getIndex(indexCont);
        Printf("secondaryContributingChannels[%d] = %d", ich - 1, secondaryContributingChannels[ich - 1]);
      }

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
        if ((labelsTPC[0].getTrackID() == labelsTOF[ilabel].getTrackID() && labelsTPC[0].getEventID() == labelsTOF[ilabel].getEventID() && labelsTPC[0].getSourceID() == labelsTOF[ilabel].getSourceID()) || (labelsITS[0].getTrackID() == labelsTOF[ilabel].getTrackID() && labelsITS[0].getEventID() == labelsTOF[ilabel].getEventID() && labelsITS[0].getSourceID() == labelsTOF[ilabel].getSourceID())) {
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
          if (labelsTPCcheck[ilabel].getTrackID() == trackIdTOF && labelsTPCcheck[ilabel].getEventID() == eventIdTOF && labelsTPCcheck[ilabel].getSourceID() == sourceIdTOF) {
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
