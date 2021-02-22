#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TFile.h"
#include "TF1.h"
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
#include "FairLogger.h"
#endif

//#define DEBUG

void checkTOFMatching(bool batchMode = true)
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
  TTree* tpcTree = (TTree*)ftracksTPC->Get("tpcrec");
  std::vector<o2::tpc::TrackTPC>* mTPCTracksArrayInp = new std::vector<o2::tpc::TrackTPC>;
  tpcTree->SetBranchAddress("TPCTracks", &mTPCTracksArrayInp);
  std::vector<o2::MCCompLabel>* mcTPC = new std::vector<o2::MCCompLabel>();
  tpcTree->SetBranchAddress("TPCTracksMCTruth", &mcTPC);

  // getting the ITS tracks
  TFile* ftracksITS = new TFile("o2trac_its.root");
  TTree* itsTree = (TTree*)ftracksITS->Get("o2sim");
  std::vector<o2::its::TrackITS>* mITSTracksArrayInp = new std::vector<o2::its::TrackITS>;
  itsTree->SetBranchAddress("ITSTrack", &mITSTracksArrayInp);
  std::vector<o2::MCCompLabel>* mcITS = new std::vector<o2::MCCompLabel>();
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
  TH1F* htofGood = new TH1F("htofGood", "tof matching |#eta|<0.9;p_{T} (GeV/c);#varepsilon", 50, 0, 5);
  TH1F* htofMism = new TH1F("htofMism", "tof matching |#eta|<0.9;p_{T} (GeV/c);#varepsilon", 50, 0, 5);
  htofMism->SetLineColor(2);
  TH1F* htrack_t = new TH1F("htrack_t", "tracks p_{T} > 0.5 GeV/c;Timestamp (#mus);N", 100, 0, 20000);
  TH1F* htof_t = new TH1F("htof_t", "tof matching p_{T} > 0.5 GeV/c, |#eta|<0.9;Timestamp (#mus);#varepsilon", 100, 0, 20000);
  TH1F* htof_tMism = new TH1F("htof_tMism", "tof matching p_{T} > 0.5 GeV/c, |#eta|<0.9;Timestamp (#mus);#varepsilon", 100, 0, 20000);
  htof_tMism->SetLineColor(2);
  TH1F* htrack_eta = new TH1F("htrack_eta", "tracks p_{T} > 0.5 GeV/c;#eta;N", 18, -0.9, 0.9);
  TH1F* htof_eta = new TH1F("htof_eta", "tof matching p_{T} > 0.5 GeV/c;#eta;#varepsilon", 18, -0.9, 0.9);
  TH1F* htof_etaMism = new TH1F("htof_etaMism", "tof matching p_{T} > 0.5 GeV/c;#eta;#varepsilon", 18, -0.9, 0.9);
  htof_etaMism->SetLineColor(2);

  TH1F* htof_res = new TH1F("htof_res", "tof residuals;residuals (cm)", 100, 0, 10);
  TH1F* htof_resMism = new TH1F("htof_resMism", "tof residuals;residuals (cm)", 100, 0, 10);
  htof_resMism->SetLineColor(2);

  TH1F* hdeltatime = new TH1F("hdeltatime", "#Deltat TOF-Track;t_{TOF} - t_{track} (#mus);N", 100, -1, 1);
  TH1F* hdeltatime_sigma = new TH1F("hdeltatime_sigma", ";#Deltat/#sigma;N", 100, -30, 30);
  TH1F* hdeltatimeMism = new TH1F("hdeltatimeMism", "#Deltat TOF-Track;t_{TOF} - t_{track} (#mus);N", 100, -1, 1);
  TH1F* hdeltatime_sigmaMism = new TH1F("hdeltatime_sigmaMism", ";#Deltat/#sigma;N", 100, -30, 30);
  hdeltatimeMism->SetLineColor(2);
  hdeltatime_sigmaMism->SetLineColor(2);

  TH2F* hchi2 = new TH2F("hchi2", "#Sum of residuals distribution;p_{T} (GeV/c);residuals (cm)", 25, 0, 5, 100, 0, 10);
  TH2F* hchi2sh = new TH2F("hchi2sh", "#Sum of residuals distribution, single hit;p_{T} (GeV/c);residuals (cm)", 25, 0, 5, 100, 0, 10);
  TH2F* hchi2dh = new TH2F("hchi2dh", "#Sum of residuals distribution, double hits;p_{T} (GeV/c);residuals (cm)", 25, 0, 5, 100, 0, 10);
  TH2F* hchi2th = new TH2F("hchi2th", "#Sum of residuals distribution, triple hits;p_{T} (GeV/c);residuals (cm)", 25, 0, 5, 100, 0, 10);

  TH1F* hMaterial = new TH1F("hMaterial", ";Material budget (X0X2 units);", 100, 0, 20);
  TProfile* hMismVsMaterial = new TProfile("hMismVsMaterial", "Mismatch fraction;Material budget (X0X2 units);Mismatch", 100, 0, 20);
  hMismVsMaterial->SetLineColor(2);
  TProfile* hResVsMaterial = new TProfile("hResVsMaterial", "Mismatch fraction;Material budget (X0X2 units);Mismatch", 100, 0, 20);
  hResVsMaterial->SetLineColor(4);

  // now looping over the entries in the matching tree
  for (int ientry = 0; ientry < matchTOF->GetEntries(); ientry++) {
    matchTOF->GetEvent(ientry);
    matchTPCITS->GetEntry(ientry);
    tpcTree->GetEntry(ientry);
    tofClTree->GetEntry(ientry);
    itsTree->GetEntry(ientry);

    // loop over tracks
    for (uint i = 0; i < mTracksArrayInp->size(); i++) {
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
    for (uint imatch = 0; imatch < TOFMatchInfo->size(); imatch++) {
      // get ITS-TPC track
      int indexITSTPCtrack = TOFMatchInfo->at(imatch).getTrackIndex();

      o2::dataformats::MatchInfoTOF infoTOF = TOFMatchInfo->at(imatch);
      int tofClIndex = infoTOF.getTOFClIndex();
      float chi2 = infoTOF.getChi2();
#ifdef DEBUG
      LOGF(INFO, "nentry in tree %d, matching %d, indexITSTPCtrack = %d, tofClIndex = %d, chi2 = %f", ientry, imatch, indexITSTPCtrack, tofClIndex, chi2);
#endif

      float matBud = infoTOF.getLTIntegralOut().getX2X0();

      //      o2::MCCompLabel label = mcTOF->getElement(mcTOF->getMCTruthHeader(tofClIndex).index);
      const auto& labelsTOF = mcTOF->getLabels(tofClIndex);
      int trackIdTOF = -1;
      int eventIdTOF = -1;
      int sourceIdTOF = -1;

      for (uint ilabel = 0; ilabel < labelsTOF.size(); ilabel++) {
#ifdef DEBUG
        LOGF(INFO, "TOF label %d: trackID = %d, eventID = %d, sourceID = %d", ilabel, labelsTOF[ilabel].getTrackID(), labelsTOF[ilabel].getEventID(), labelsTOF[ilabel].getSourceID());
#endif
        if (ilabel == 0) {
          trackIdTOF = labelsTOF[ilabel].getTrackID();
          eventIdTOF = labelsTOF[ilabel].getEventID();
          sourceIdTOF = labelsTOF[ilabel].getSourceID();
        }
      }
      o2::tof::Cluster tofCluster = mTOFClustersArrayInp->at(tofClIndex);
      int nContributingChannels = tofCluster.getNumOfContributingChannels();
      int mainContributingChannel = tofCluster.getMainContributingChannel();
#ifdef DEBUG
      LOGF(INFO, "The TOF cluster has %d contributing channels, and the main one is %d", nContributingChannels, mainContributingChannel);
#endif
      int* indices = new int();
      o2::tof::Geo::getVolumeIndices(mainContributingChannel, indices);
#ifdef DEBUG
      LOGF(INFO, "Indices of main contributing channel are %d, %d, %d, %d, %d", indices[0], indices[1], indices[2], indices[3], indices[4]);
#endif
      bool isUpLeft = tofCluster.isAdditionalChannelSet(o2::tof::Cluster::kUpLeft);
      bool isUp = tofCluster.isAdditionalChannelSet(o2::tof::Cluster::kUp);
      bool isUpRight = tofCluster.isAdditionalChannelSet(o2::tof::Cluster::kUpRight);
      bool isRight = tofCluster.isAdditionalChannelSet(o2::tof::Cluster::kRight);
      bool isDownRight = tofCluster.isAdditionalChannelSet(o2::tof::Cluster::kDownRight);
      bool isDown = tofCluster.isAdditionalChannelSet(o2::tof::Cluster::kDown);
      bool isDownLeft = tofCluster.isAdditionalChannelSet(o2::tof::Cluster::kDownLeft);
      bool isLeft = tofCluster.isAdditionalChannelSet(o2::tof::Cluster::kLeft);
#ifdef DEBUG
      LOGF(INFO, "isUpLeft = %d, isUp = %d, isUpRight = %d, isRight = %d, isDownRight = %d, isDown = %d, isDownLeft = %d, isLeft = %d", isUpLeft, isUp, isUpRight, isRight, isDownRight, isDown, isDownLeft, isLeft);
#endif
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
#ifdef DEBUG
        LOGF(INFO, "secondaryContributingChannel[down] = %d", secondaryContributingChannel);
#endif
        indexCont[3] = indices[3];
      }
      if (isDownRight) {
        indexCont[3]--;
        indexCont[4]++;
        numberOfSecondaryContributingChannels++;
        secondaryContributingChannel = o2::tof::Geo::getIndex(indexCont);
#ifdef DEBUG
        LOGF(INFO, "secondaryContributingChannel[downright] = %d", secondaryContributingChannel);
#endif
        indexCont[3] = indices[3];
        indexCont[4] = indices[4];
      }
      if (isDownLeft) {
        indexCont[3]--;
        indexCont[4]--;
        numberOfSecondaryContributingChannels++;
        secondaryContributingChannel = o2::tof::Geo::getIndex(indexCont);
#ifdef DEBUG
        LOGF(INFO, "secondaryContributingChannel[downleft] = %d", secondaryContributingChannel);
#endif
        indexCont[3] = indices[3];
        indexCont[4] = indices[4];
      }
      if (isUp) {
        indexCont[3]++;
        numberOfSecondaryContributingChannels++;
        secondaryContributingChannel = o2::tof::Geo::getIndex(indexCont);
#ifdef DEBUG
        LOGF(INFO, "secondaryContributingChannel[up] = %d", secondaryContributingChannel);
#endif
        indexCont[3] = indices[3];
      }
      if (isUpRight) {
        indexCont[3]++;
        indexCont[4]++;
        numberOfSecondaryContributingChannels++;
        secondaryContributingChannel = o2::tof::Geo::getIndex(indexCont);
#ifdef DEBUG
        LOGF(INFO, "secondaryContributingChannel[upright] = %d", secondaryContributingChannel);
#endif
        indexCont[3] = indices[3];
        indexCont[4] = indices[4];
      }
      if (isUpLeft) { // increase padZ
        indexCont[3]++;
        indexCont[4]--;
        numberOfSecondaryContributingChannels++;
        secondaryContributingChannel = o2::tof::Geo::getIndex(indexCont);
#ifdef DEBUG
        LOGF(INFO, "secondaryContributingChannel[upleft] = %d", secondaryContributingChannel);
#endif
        indexCont[3] = indices[3];
        indexCont[4] = indices[4];
      }
      if (isRight) { // increase padX
        indexCont[4]++;
        numberOfSecondaryContributingChannels++;
        secondaryContributingChannel = o2::tof::Geo::getIndex(indexCont);
#ifdef DEBUG
        LOGF(INFO, "secondaryContributingChannel[right] = %d", secondaryContributingChannel);
#endif
        indexCont[4] = indices[4];
      }
      if (isLeft) { // decrease padX
        indexCont[4]--;
        numberOfSecondaryContributingChannels++;
        secondaryContributingChannel = o2::tof::Geo::getIndex(indexCont);
#ifdef DEBUG
        LOGF(INFO, "secondaryContributingChannel[left] = %d", secondaryContributingChannel);
#endif
        indexCont[4] = indices[4];
      }
#ifdef DEBUG
      LOGF(INFO, "Total number of secondary channels= %d", numberOfSecondaryContributingChannels);
#endif
      o2::dataformats::TrackTPCITS trackITSTPC = mTracksArrayInp->at(indexITSTPCtrack);

      const auto evIdxTPC = trackITSTPC.getRefTPC();
#ifdef DEBUG
      LOG(INFO) << "matched TPCtrack index:", evIdxTPC;
#endif
      const auto evIdxITS = trackITSTPC.getRefITS();
#ifdef DEBUG
      LOG(INFO) << "matched ITStrack index: ", evIdxITS;
#endif
      // getting the TPC labels
      const auto& labelsTPC = (*mcTPC)[evIdxTPC];
#ifdef DEBUG
      LOGF(INFO, "TPC label: trackID = %d, eventID = %d, sourceID = %d", labelsTPC.getTrackID(), labelsTPC.getEventID(), labelsTPC.getSourceID());
#endif

      // getting the ITS labels
      const auto& labelsITS = (*mcITS)[evIdxITS];
#ifdef DEBUG
      LOGF(INFO, "ITS label: trackID = %d, eventID = %d, sourceID = %d", labelsITS.getTrackID(), labelsITS.getEventID(), labelsITS.getSourceID());
#endif
      bool bMatched = kFALSE;
      for (uint ilabel = 0; ilabel < labelsTOF.size(); ilabel++) {
        if ((abs(labelsTPC.getTrackID()) == labelsTOF[ilabel].getTrackID() && labelsTPC.getEventID() == labelsTOF[ilabel].getEventID() && labelsTPC.getSourceID() == labelsTOF[ilabel].getSourceID()) || (labelsITS.getTrackID() == labelsTOF[ilabel].getTrackID() && labelsITS.getEventID() == labelsTOF[ilabel].getEventID() && labelsITS.getSourceID() == labelsTOF[ilabel].getSourceID())) {
          nGoodMatches++;
          bMatched = kTRUE;
          break;
        }
      }
      if (!bMatched)
        nBadMatches++;

      hMaterial->Fill(matBud);
      hMismVsMaterial->Fill(matBud, !bMatched);
      if (bMatched)
        hResVsMaterial->Fill(matBud, chi2);

      if (bMatched)
        htof_res->Fill(chi2);
      else
        htof_resMism->Fill(chi2);

      float deltatime = tofCluster.getTime() * 1E-6 - trackITSTPC.getTimeMUS().getTimeStamp(); // in mus
      if (bMatched) {
        hdeltatime->Fill(deltatime);
        hdeltatime_sigma->Fill(deltatime / trackITSTPC.getTimeMUS().getTimeStampError());
        hchi2->Fill(trackITSTPC.getPt(), TOFMatchInfo->at(imatch).getChi2());
      } else {
        hdeltatimeMism->Fill(deltatime);
        hdeltatime_sigmaMism->Fill(deltatime / trackITSTPC.getTimeMUS().getTimeStampError());
      }
      if (bMatched) {
        if (nContributingChannels == 1)
          hchi2sh->Fill(trackITSTPC.getPt(), TOFMatchInfo->at(imatch).getChi2());
        else if (nContributingChannels == 2)
          hchi2dh->Fill(trackITSTPC.getPt(), TOFMatchInfo->at(imatch).getChi2());
        else if ((isRight || isLeft) && (isUp || isDown)) {
          hchi2th->Fill(trackITSTPC.getPt(), TOFMatchInfo->at(imatch).getChi2());
        }
      }

      if (TMath::Abs(trackITSTPC.getEta()) < 0.9) {
        htof->Fill(trackITSTPC.getPt());
        if (bMatched)
          htofGood->Fill(trackITSTPC.getPt());
        else
          htofMism->Fill(trackITSTPC.getPt());
        if (trackITSTPC.getPt() > 0.5) {
          if (bMatched)
            htof_t->Fill(trackITSTPC.getTimeMUS().getTimeStamp());
          else
            htof_tMism->Fill(trackITSTPC.getTimeMUS().getTimeStamp());
          if (bMatched)
            htof_eta->Fill(trackITSTPC.getEta());
          else
            htof_etaMism->Fill(trackITSTPC.getEta());
        }
      }

      bool TPCfound = false;
      bool ITSfound = false;
      for (uint i = 0; i < mTracksArrayInp->size(); i++) {
        o2::dataformats::TrackTPCITS trackITSTPC = mTracksArrayInp->at(i);
        const auto idxTPCcheck = trackITSTPC.getRefTPC();
        const auto idxITScheck = trackITSTPC.getRefITS();
        const auto& labelsTPCcheck = (*mcTPC)[idxTPCcheck.getIndex()];
        if (abs(labelsTPCcheck.getTrackID()) == trackIdTOF && labelsTPCcheck.getEventID() == eventIdTOF && labelsTPCcheck.getSourceID() == sourceIdTOF) {
#ifdef DEBUG
          LOGF(INFO, "The TPC track that should have been matched to TOF is number %d", i);
#endif
          TPCfound = true;
        }
        const auto& labelsITScheck = (*mcITS)[idxITScheck.getIndex()];
        if (labelsITScheck.getTrackID() == trackIdTOF && labelsITScheck.getEventID() == eventIdTOF && labelsITScheck.getSourceID() == sourceIdTOF) {
#ifdef DEBUG
          LOGF(INFO, "The ITS track that should have been matched to TOF is number %d", i);
#endif
          ITSfound = true;
        }
      }
#ifdef DEBUG
      if (!TPCfound)
        LOGF(INFO, "There is no TPC track found that should have corresponded to this TOF cluster!");
      if (!ITSfound)
        LOGF(INFO, "There is no ITS track found that should have corresponded to this TOF cluster!");
#endif
    }
  }

  new TCanvas;

  LOGF(INFO, "Number of      matches = %d", nMatches);
  LOGF(INFO, "Number of GOOD matches = %d (%.2f)", nGoodMatches, (float)nGoodMatches / nMatches);
  LOGF(INFO, "Number of BAD  matches = %d (%.2f)", nBadMatches, (float)nBadMatches / nMatches);

  TFile* fout = nullptr;
  if (batchMode)
    fout = new TFile("tofmatching_qa.root", "RECREATE");

  htofMism->Divide(htofMism, htof, 1, 1, "B");
  htof->Divide(htofGood, htrack, 1, 1, "B");
  htof->SetMarkerStyle(20);
  if (batchMode) {
    htof->Write();
    htofMism->Write();
  } else {
    htof->Draw("P");
    htofMism->Draw("SAME");
  }

  htof_eta->Divide(htof_eta, htrack_eta, 1, 1, "B");
  htof_etaMism->Divide(htof_etaMism, htrack_eta, 1, 1, "B");
  htof_eta->SetMarkerStyle(20);
  if (batchMode) {
    htof_eta->Write();
    htof_etaMism->Write();
  } else {
    new TCanvas;
    htof_eta->Draw("P");
    htof_etaMism->Draw("same");
  }

  htof_t->Divide(htof_t, htrack_t, 1, 1, "B");
  htof_tMism->Divide(htof_tMism, htrack_t, 1, 1, "B");
  htof_t->SetMarkerStyle(20);
  if (batchMode) {
    htof_t->Write();
    htof_tMism->Write();
    hdeltatime->Write();
    hdeltatimeMism->Write();
    hdeltatime_sigma->Write();
    hdeltatime_sigmaMism->Write();
  } else {
    new TCanvas;
    htof_t->Draw("P");
    htof_tMism->Draw("same");
    new TCanvas;
    hdeltatime->Draw();
    hdeltatimeMism->Draw("SAME");

    new TCanvas;
    hdeltatime_sigma->Draw();
    hdeltatime_sigmaMism->Draw("SAME");
  }
  if (batchMode) {
    hchi2->Write();
    hchi2sh->Write();
    hchi2dh->Write();
    hchi2th->Write();
  } else {
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
  }

  if (batchMode) {
    htof_res->Write();
    htof_resMism->Write();

    hMaterial->Write();
    hMismVsMaterial->Write();
    hResVsMaterial->Write();
  } else {
    TCanvas* cresiduals = new TCanvas("cresiduals", "cresiduals");
    htof_res->Draw();
    htof_resMism->Draw("SAME");

    TCanvas* cmaterial = new TCanvas("cmaterial", "cmaterial");
    hMaterial->DrawNormalized("", 10);
    hMismVsMaterial->Draw("SAME");
    hResVsMaterial->Draw("SAME");
  }

  float fraction = hchi2dh->GetEntries() * 1. / hchi2->GetEntries();
  float fractionErr = TMath::Sqrt(fraction * (1 - fraction) / hchi2->GetEntries());
  LOGF(INFO, "Fraction of multiple hits = (%.1f +/- %.1f)%c", fraction * 100, fractionErr * 100, '%');

  htof->Fit("pol0", "", "", 1, 5);
  float effMatch = 0;
  if (htof->GetListOfFunctions()->At(0))
    effMatch = ((TF1*)htof->GetListOfFunctions()->At(0))->GetParameter(0);
  float effMatchErr = 0;
  if (htof->GetListOfFunctions()->At(0))
    effMatchErr = ((TF1*)htof->GetListOfFunctions()->At(0))->GetParError(0);
  LOGF(INFO, "TOF matching eff (pt > 1) = %f +/- %f", effMatch, effMatchErr);

  htofMism->Fit("pol0", "", "", 1, 5);
  float mismMatch = 0;
  if (htofMism->GetListOfFunctions()->At(0))
    mismMatch = ((TF1*)htofMism->GetListOfFunctions()->At(0))->GetParameter(0);
  float mismMatchErr = 0;
  if (htofMism->GetListOfFunctions()->At(0))
    mismMatchErr = ((TF1*)htofMism->GetListOfFunctions()->At(0))->GetParError(0);
  LOGF(INFO, "TOF-track mismatch (pt > 1) = %f +/- %f", mismMatch, mismMatchErr);

  if (fout)
    fout->Close();

  return;
}
