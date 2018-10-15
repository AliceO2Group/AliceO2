void checkTOFMatching(){

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
  TTree* tpcTree = (TTree*) ftracksTPC->Get("events");
  std::vector<o2::TPC::TrackTPC>* mTPCTracksArrayInp = new std::vector<o2::TPC::TrackTPC>;
  tpcTree->SetBranchAddress("TPCTracks", &mTPCTracksArrayInp);
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mcTPC = new o2::dataformats::MCTruthContainer<o2::MCCompLabel>();
  tpcTree->SetBranchAddress("TPCTracksMCTruth", &mcTPC);

  // getting the ITS tracks
  TFile* ftracksITS = new TFile("o2trac_its.root");
  TTree* itsTree = (TTree*) ftracksITS->Get("o2sim");
  std::vector<o2::ITS::TrackITS>* mITSTracksArrayInp = new std::vector<o2::ITS::TrackITS>;
  itsTree->SetBranchAddress("ITSTrack", &mITSTracksArrayInp);
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mcITS = new o2::dataformats::MCTruthContainer<o2::MCCompLabel>();
  itsTree->SetBranchAddress("ITSTrackMCTruth", &mcITS);

  // getting the TOF clusters
  TFile* fclustersTOF = new TFile("tofclusters.root");
  TTree* tofClTree = (TTree*) fclustersTOF->Get("o2sim");
  std::vector<o2::tof::Cluster>* mTOFClustersArrayInp = new std::vector<o2::tof::Cluster>;
  tofClTree->SetBranchAddress("TOFCluster", &mTOFClustersArrayInp);
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mcTOF = new o2::dataformats::MCTruthContainer<o2::MCCompLabel>();
  tofClTree->SetBranchAddress("TOFClusterMCTruth", &mcTOF);
  
  tpcTree->GetEntry(0);
  tofClTree->GetEntry(0);
  
  // now looping over the entries in the matching tree
  for (int ientry = 0; ientry < matchTOF->GetEntries(); ientry++){
    matchTOF->GetEvent(ientry);
    matchTPCITS->GetEntry(ientry);
    // now looping over the matched tracks
    for (int imatch = 0; imatch < TOFMatchInfo->size(); imatch++){
      int indexITSTPCtrack = TOFMatchInfo->at(imatch).first;
      o2::dataformats::MatchInfoTOF infoTOF = TOFMatchInfo->at(imatch).second;
      int tofClIndex = infoTOF.getTOFClIndex();
      float chi2 = infoTOF.getChi2();
      Printf("\nentry in tree %d, matching %d, indexITSTPCtrack = %d, tofClIndex = %d, chi2 = %f", ientry, imatch, indexITSTPCtrack, tofClIndex, chi2);
      //      o2::MCCompLabel label = mcTOF->getElement(mcTOF->getMCTruthHeader(tofClIndex).index);
      const auto& labelsTOF = mcTOF->getLabels(tofClIndex);
      for (int ilabel = 0; ilabel < labelsTOF.size(); ilabel++){
	Printf("TOF label %d: trackID = %d, eventID = %d, sourceID = %d", ilabel, labelsTOF[ilabel].getTrackID(), labelsTOF[ilabel].getEventID(), labelsTOF[ilabel].getSourceID());
      }
      o2::dataformats::TrackTPCITS trackITSTPC = mTracksArrayInp->at(indexITSTPCtrack);
      const o2::dataformats::EvIndex<int, int> &evIdxTPC = trackITSTPC.getRefTPC();
      Printf("matched TPCtrack: eventID = %d, indexID = %d", evIdxTPC.getEvent(), evIdxTPC.getIndex());
      const o2::dataformats::EvIndex<int, int> &evIdxITS = trackITSTPC.getRefITS();
      Printf("matched ITStrack: eventID = %d, indexID = %d", evIdxITS.getEvent(), evIdxITS.getIndex());
      itsTree->GetEntry(evIdxITS.getEvent());

      // getting the TPC labels
      const auto& labelsTPC = mcTPC->getLabels(evIdxTPC.getIndex());
      for (int ilabel = 0; ilabel < labelsTPC.size(); ilabel++){
	Printf("TPC label %d: trackID = %d, eventID = %d, sourceID = %d", ilabel, labelsTPC[ilabel].getTrackID(), labelsTPC[ilabel].getEventID(), labelsTPC[ilabel].getSourceID());
      }

      // getting the ITS labels
      const auto& labelsITS = mcITS->getLabels(evIdxITS.getIndex());
      for (int ilabel = 0; ilabel < labelsITS.size(); ilabel++){
	Printf("ITS label %d: trackID = %d, eventID = %d, sourceID = %d", ilabel, labelsITS[ilabel].getTrackID(), labelsITS[ilabel].getEventID(), labelsITS[ilabel].getSourceID());
      }

    }

  }

  

  
  return;
}

  
  
