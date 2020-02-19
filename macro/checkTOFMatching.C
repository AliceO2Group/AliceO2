#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TLine.h"
#include "TProfile.h"
#include "TMath.h"
#include "TCanvas.h"
#include "GlobalTracking/MatchTOF.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsTOF/Cluster.h"
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

  int nMatches = 0;
  int nGoodMatches = 0;
  int nBadMatches = 0;

  TH1F* htrack = new TH1F("htrack", "tracks;p_{T} (GeV/c);N", 50, 0, 5);
  TH1F* htof = new TH1F("htof", "tof matching |#eta|<0.9;p_{T} (GeV/c);#varepsilon", 50, 0, 5);
  TH1F* htrack_t = new TH1F("htrack_t", "tracks p_{T} > 0.5 GeV/c;Timestamp (#mus);N", 100, 0, 20000);
  TH1F* htof_t = new TH1F("htof_t", "tof matching p_{T} > 0.5 GeV/c, |#eta|<0.9;Timestamp (#mus);#varepsilon", 100, 0, 20000);
  TH1F* htrack_eta = new TH1F("htrack_eta", "tracks p_{T} > 0.5 GeV/c;#eta;N", 18, -0.9, 0.9);
  TH1F* htof_eta = new TH1F("htof_eta", "tof matching p_{T} > 0.5 GeV/c;#eta;#varepsilon", 18, -0.9, 0.9);

  TH1F* hdeltatime = new TH1F("hdeltatime", "#Deltat TOF-Track;t_{TOF} - t_{track} (#mus);N", 100, -1, 1);
  TH1F* hdeltatime_sigma = new TH1F("hdeltatime_sigma", ";#Deltat/#sigma;N", 100, -30, 30);

  TH2F* hchi2 = new TH2F("hchi2", "#Sum of residuals distribution;p_{T} (GeV/c);residuals (cm)", 25, 0, 5, 100, 0, 10);
  TH2F* hchi2sh = new TH2F("hchi2sh", "#Sum of residuals distribution, single hit;p_{T} (GeV/c);residuals (cm)", 25, 0, 5, 100, 0, 10);
  TH2F* hchi2dh = new TH2F("hchi2dh", "#Sum of residuals distribution, double hits;p_{T} (GeV/c);residuals (cm)", 25, 0, 5, 100, 0, 10);
  TH2F* hchi2th = new TH2F("hchi2th", "#Sum of residuals distribution, triple hits;p_{T} (GeV/c);residuals (cm)", 25, 0, 5, 100, 0, 10);

  // now looping over the entries in the matching tree
  for (int ientry = 0; ientry < matchTOF->GetEntries(); ientry++) {
    matchTOF->GetEvent(ientry);
    matchTPCITS->GetEntry(ientry);
    tpcTree->GetEntry(ientry);
    tofClTree->GetEntry(ientry);
    itsTree->GetEntry(ientry);

    // loop over tracks
    for (int i = 0; i < mTracksArrayInp->size(); i++) {
      o2::dataformats::TrackTPCITS trackITSTPC = mTracksArrayInp->at(i);
      if (TMath::Abs(trackITSTPC.getEta()) < 0.9) {
        htrack->Fill(trackITSTPC.getPt());
        if (trackITSTPC.getPt() > 0.5) {
          htrack_t->Fill(trackITSTPC.getTimeMUS().getTimeStamp());
          htrack_eta->Fill(trackITSTPC.getEta());
        }
      }
    }
    // now looping over the matched tracks
    nMatches += TOFMatchInfo->size();
    for (int imatch = 0; imatch < TOFMatchInfo->size(); imatch++) {
      // get ITS-TPC track
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

      float deltatime = tofCluster.getTime() * 1E-6 - trackITSTPC.getTimeMUS().getTimeStamp(); // in mus
      hdeltatime->Fill(deltatime);
      hdeltatime_sigma->Fill(deltatime / trackITSTPC.getTimeMUS().getTimeStampError());
      hchi2->Fill(trackITSTPC.getPt(), TOFMatchInfo->at(imatch).getChi2());
      if (nContributingChannels == 1)
        hchi2sh->Fill(trackITSTPC.getPt(), TOFMatchInfo->at(imatch).getChi2());
      else
        hchi2dh->Fill(trackITSTPC.getPt(), TOFMatchInfo->at(imatch).getChi2());
      if (tofCluster.getSigmaY2() < 0.2 && tofCluster.getSigmaZ2() < 0.2)
        hchi2th->Fill(trackITSTPC.getPt(), TOFMatchInfo->at(imatch).getChi2());

      if (TMath::Abs(trackITSTPC.getEta()) < 0.9) {
        htof->Fill(trackITSTPC.getPt());
        if (trackITSTPC.getPt() > 0.5) {
          htof_t->Fill(trackITSTPC.getTimeMUS().getTimeStamp());
          htof_eta->Fill(trackITSTPC.getEta());
        }
      }
      const auto evIdxTPC = trackITSTPC.getRefTPC();
      Printf("matched TPCtrack index = %d", evIdxTPC);
      const auto evIdxITS = trackITSTPC.getRefITS();
      Printf("matched ITStrack index = %d", evIdxITS);

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
        const auto evIdxTPCcheck = trackITSTPC.getRefTPC();
        const auto evIdxITScheck = trackITSTPC.getRefITS();
        const auto& labelsTPCcheck = mcTPC->getLabels(evIdxTPCcheck);
        for (int ilabel = 0; ilabel < labelsTPCcheck.size(); ilabel++) {
          if (abs(labelsTPCcheck[ilabel].getTrackID()) == trackIdTOF && labelsTPCcheck[ilabel].getEventID() == eventIdTOF && labelsTPCcheck[ilabel].getSourceID() == sourceIdTOF) {
            Printf("The TPC track that should have been matched to TOF is number %d", i);
            TPCfound = true;
          }
        }
        const auto& labelsITScheck = mcITS->getLabels(evIdxITScheck);
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

  htof->Divide(htof, htrack, 1, 1, "B");
  htof->SetMarkerStyle(20);
  htof->Draw("P");

  new TCanvas;
  htof_eta->Divide(htof_eta, htrack_eta, 1, 1, "B");
  htof_eta->SetMarkerStyle(20);
  htof_eta->Draw("P");

  new TCanvas;
  htof_t->Divide(htof_t, htrack_t, 1, 1, "B");
  htof_t->SetMarkerStyle(20);
  htof_t->Draw("P");

  new TCanvas;
  hdeltatime->Draw();

  new TCanvas;
  hdeltatime_sigma->Draw();

  TCanvas* cres = new TCanvas();
  cres->Divide(2, 2);
  cres->cd(1)->SetLogz();
  hchi2->Draw("colz");
  hchi2->ProfileX()->Draw("same");
  TLine* l = new TLine(0, 0.983575, 5, 0.983575);
  l->Draw("SAME");
  l->SetLineStyle(2);
  l->SetLineWidth(2);
  l->SetLineColor(4);
  cres->cd(2)->SetLogz();
  hchi2sh->Draw("colz");
  hchi2sh->ProfileX()->Draw("same");
  TLine* l2 = new TLine(0, 1.044939, 5, 1.044939);
  l2->Draw("SAME");
  l2->SetLineStyle(2);
  l2->SetLineWidth(2);
  l2->SetLineColor(4);
  cres->cd(3)->SetLogz();
  hchi2dh->Draw("colz");
  hchi2dh->ProfileX()->Draw("same");
  TLine* l3 = new TLine(0, 0.73811975, 5, 0.73811975);
  l3->Draw("SAME");
  l3->SetLineStyle(2);
  l3->SetLineWidth(2);
  l3->SetLineColor(4);
  cres->cd(4)->SetLogz();
  hchi2th->Draw("colz");
  hchi2th->ProfileX()->Draw("same");
  TLine* l4 = new TLine(0, 0.3, 5, 0.3);
  l4->Draw("SAME");
  l4->SetLineStyle(2);
  l4->SetLineWidth(2);
  l4->SetLineColor(4);

  float fraction = hchi2dh->GetEntries() * 1. / hchi2->GetEntries();
  float fractionErr = TMath::Sqrt(fraction * (1 - fraction) / hchi2->GetEntries());
  printf("Fraction of multiple hits = (%.1f +/- %.1f)\%\n", fraction * 100, fractionErr * 100);

  return;
}
