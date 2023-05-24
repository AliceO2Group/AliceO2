// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "GlobalTracking/MatchGlobalFwdAssessment.h"
#include "Framework/InputSpec.h"
#include "DetectorsBase/GeometryManager.h"
#include <Framework/InputRecord.h>
#include <TPaveText.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TFile.h>
#include <TGraph.h>

using namespace o2::globaltracking;

//__________________________________________________________
void GloFwdAssessment::init(bool finalizeAnalysis)
{
  mFinalizeAnalysis = finalizeAnalysis;
  createHistos();
}

//__________________________________________________________
void GloFwdAssessment::reset()
{

  mTrackNumberOfClusters->Reset();
  mTrackInvQPt->Reset();
  mTrackChi2->Reset();
  mTrackCharge->Reset();
  mTrackPhi->Reset();
  mTrackEta->Reset();
  for (auto minNClusters : sMinNClustersList) {
    auto nHisto = minNClusters - sMinNClustersList[0];
    mTrackEtaNCls[nHisto]->Reset();
    mTrackPhiNCls[nHisto]->Reset();
    mTrackXYNCls[nHisto]->Reset();
    mTrackEtaPhiNCls[nHisto]->Reset();
  }

  mTrackCotl->Reset();

  if (mUseMC) {
    mPairables.clear();
    mTrueTracksMap.clear();
    mPairableTracksMap.clear();

    mHistPhiRecVsPhiGen->Reset();
    mHistEtaRecVsEtaGen->Reset();
    for (int trackType = 0; trackType < kNumberOfTrackTypes; trackType++) {
      mHistPhiVsEta[trackType]->Reset();
      mHistPtVsEta[trackType]->Reset();
      mHistPhiVsPt[trackType]->Reset();
      mHistZvtxVsEta[trackType]->Reset();
      if (trackType == kGen || trackType == kPairable) {
        mHistRVsZ[trackType]->Reset();
      }
    }

    auto hC = mChargeMatchEff->GetCopyTotalHisto();
    hC->Reset();
    mChargeMatchEff->SetTotalHistogram(*hC, "");
    mChargeMatchEff->SetPassedHistogram(*hC, "");

    mPairingEtaPt->Reset();
    mTruePairingEtaPt->Reset();
    for (auto& h : mTH3Histos) {
      h->Reset();
    }
  }
}

//__________________________________________________________
void GloFwdAssessment::createHistos()
{

  // Creating data-only histos
  mTrackNumberOfClusters = std::make_unique<TH1F>("mGlobalFwdNumberOfMFTClusters",
                                                  "Number Of Clusters Per Track; # clusters; # entries", 10, 0.5, 10.5);

  mTrackInvQPt = std::make_unique<TH1F>("mGlobalFwdInvQPt", "Track q/p_{T}; q/p_{T} [1/GeV]; # entries", 100, -5, 5);

  mTrackChi2 = std::make_unique<TH1F>("mGlobalFwdChi2", "Track #chi^{2}; #chi^{2}; # entries", 202, -0.5, 100.5);

  mTrackCharge = std::make_unique<TH1F>("mGlobalFwdCharge", "Track Charge; q; # entries", 3, -1.5, 1.5);

  mTrackPhi = std::make_unique<TH1F>("mGlobalFwdPhi", "Track #phi; #phi; # entries", 100, -3.2, 3.2);

  mTrackEta = std::make_unique<TH1F>("mGlobalFwdEta", "Track #eta; #eta; # entries", 50, -4, -2);

  for (auto minNClusters : sMinNClustersList) {
    auto nHisto = minNClusters - sMinNClustersList[0];
    mTrackEtaNCls[nHisto] = std::make_unique<TH1F>(Form("mGlobalFwdEta_%d_MinClusters", minNClusters), Form("Track #eta (NCls >= %d); #eta; # entries", minNClusters), 50, -4, -2);

    mTrackPhiNCls[nHisto] = std::make_unique<TH1F>(Form("mGlobalFwdPhi_%d_MinClusters", minNClusters), Form("Track #phi (NCls >= %d); #phi; # entries", minNClusters), 100, -3.2, 3.2);

    mTrackXYNCls[nHisto] = std::make_unique<TH2F>(Form("mGlobalFwdXY_%d_MinClusters", minNClusters), Form("Track Position (NCls >= %d); x; y", minNClusters), 320, -16, 16, 320, -16, 16);
    mTrackXYNCls[nHisto]->SetOption("COLZ");

    mTrackEtaPhiNCls[nHisto] = std::make_unique<TH2F>(Form("mGlobalFwdEtaPhi_%d_MinClusters", minNClusters), Form("Track #eta , #phi (NCls >= %d); #eta; #phi", minNClusters), 50, -4, -2, 100, -3.2, 3.2);
    mTrackEtaPhiNCls[nHisto]->SetOption("COLZ");
  }

  mTrackCotl = std::make_unique<TH1F>("mGlobalFwdCotl", "Track cot #lambda; cot #lambda; # entries", 100, -0.25, 0);

  // Creating MC-based histos
  if (mUseMC) {

    LOG(info) << "Initializing MC Reader";
    if (!mcReader.initFromDigitContext("collisioncontext.root")) {
      throw std::invalid_argument("initialization of MCKinematicsReader failed");
    }

    mHistPhiRecVsPhiGen = std::make_unique<TH2F>("mGMTrackPhiRecVsPhiGen", "Phi Rec Vs Phi Gen of true reco tracks ", 24, -TMath::Pi(), TMath::Pi(), 24, -TMath::Pi(), TMath::Pi());
    mHistPhiRecVsPhiGen->SetXTitle((std::string("#phi of ") + mNameOfTrackTypes[kGen]).c_str());
    mHistPhiRecVsPhiGen->SetYTitle((std::string("#phi of ") + mNameOfTrackTypes[kRecoTrue]).c_str());
    mHistPhiRecVsPhiGen->Sumw2();
    mHistPhiRecVsPhiGen->SetOption("COLZ");

    mHistEtaRecVsEtaGen = std::make_unique<TH2F>("mGMTrackEtaRecVsEtaGen", "Eta Rec Vs Eta Gen of true reco tracks ", 35, 1.0, 4.5, 35, 1.0, 4.5);
    mHistEtaRecVsEtaGen->SetXTitle((std::string("#eta of ") + mNameOfTrackTypes[kGen]).c_str());
    mHistEtaRecVsEtaGen->SetYTitle((std::string("#eta of ") + mNameOfTrackTypes[kRecoTrue]).c_str());
    mHistEtaRecVsEtaGen->Sumw2();
    mHistEtaRecVsEtaGen->SetOption("COLZ");

    for (int trackType = 0; trackType < kNumberOfTrackTypes; trackType++) {
      // mHistPhiVsEta
      mHistPhiVsEta[trackType] = std::make_unique<TH2F>((std::string("mGMTrackPhiVsEta") + mNameOfTrackTypes[trackType]).c_str(), (std::string("Phi Vs Eta of ") + mNameOfTrackTypes[trackType]).c_str(), 35, 1.0, 4.5, 24, -TMath::Pi(), TMath::Pi());
      mHistPhiVsEta[trackType]->SetXTitle((std::string("#eta of ") + mNameOfTrackTypes[trackType]).c_str());
      mHistPhiVsEta[trackType]->SetYTitle((std::string("#phi of ") + mNameOfTrackTypes[trackType]).c_str());
      mHistPhiVsEta[trackType]->Sumw2();
      mHistPhiVsEta[trackType]->SetOption("COLZ");

      // mHistPtVsEta
      mHistPtVsEta[trackType] = std::make_unique<TH2F>((std::string("mGMTrackPtVsEta") + mNameOfTrackTypes[trackType]).c_str(), (std::string("Pt Vs Eta of ") + mNameOfTrackTypes[trackType]).c_str(), 35, 1.0, 4.5, 40, 0., 10.);
      mHistPtVsEta[trackType]->SetXTitle((std::string("#eta of ") + mNameOfTrackTypes[trackType]).c_str());
      mHistPtVsEta[trackType]->SetYTitle((std::string("p_{T} (GeV/c) of ") + mNameOfTrackTypes[trackType]).c_str());
      mHistPtVsEta[trackType]->Sumw2();
      mHistPtVsEta[trackType]->SetOption("COLZ");

      // mHistPhiVsPt
      mHistPhiVsPt[trackType] = std::make_unique<TH2F>((std::string("mGMTrackPhiVsPt") + mNameOfTrackTypes[trackType]).c_str(), (std::string("Phi Vs Pt of ") + mNameOfTrackTypes[trackType]).c_str(), 40, 0., 10., 24, -TMath::Pi(), TMath::Pi());
      mHistPhiVsPt[trackType]->SetXTitle((std::string("p_{T} (GeV/c) of ") + mNameOfTrackTypes[trackType]).c_str());
      mHistPhiVsPt[trackType]->SetYTitle((std::string("#phi of ") + mNameOfTrackTypes[trackType]).c_str());
      mHistPhiVsPt[trackType]->Sumw2();
      mHistPhiVsPt[trackType]->SetOption("COLZ");

      if (trackType != kReco) {
        // mHistZvtxVsEta
        mHistZvtxVsEta[trackType] = std::make_unique<TH2F>((std::string("mGMTrackZvtxVsEta") + mNameOfTrackTypes[trackType]).c_str(), (std::string("Z_{vtx} Vs Eta of ") + mNameOfTrackTypes[trackType]).c_str(), 35, 1.0, 4.5, 15, -15, 15);
        mHistZvtxVsEta[trackType]->SetXTitle((std::string("#eta of ") + mNameOfTrackTypes[trackType]).c_str());
        mHistZvtxVsEta[trackType]->SetYTitle((std::string("z_{vtx} (cm) of ") + mNameOfTrackTypes[trackType]).c_str());
        mHistZvtxVsEta[trackType]->Sumw2();
        mHistZvtxVsEta[trackType]->SetOption("COLZ");
      }
      // mHistRVsZ]
      if (trackType == kGen || trackType == kPairable) {
        mHistRVsZ[trackType] = std::make_unique<TH2F>((std::string("mGMTrackRVsZ") + mNameOfTrackTypes[trackType]).c_str(), (std::string("R Vs Z of ") + mNameOfTrackTypes[trackType]).c_str(), 400, -80., 20., 400, 0., 80.);
        mHistRVsZ[trackType]->SetXTitle((std::string("z (cm) origin of ") + mNameOfTrackTypes[trackType]).c_str());
        mHistRVsZ[trackType]->SetYTitle((std::string("R (cm) radius of origin of ") + mNameOfTrackTypes[trackType]).c_str());
        mHistRVsZ[trackType]->Sumw2();
        mHistRVsZ[trackType]->SetOption("COLZ");
      }
    }

    // Histos for Reconstruction assessment

    mChargeMatchEff = std::make_unique<TEfficiency>("mGMTrackQMatchEff", "Charge Match;p_t [GeV];#epsilon", 50, 0, 20);

    const int nTH3Histos = TH3Names.size();
    auto n3Histo = 0;
    for (auto& h : mTH3Histos) {
      h = std::make_unique<TH3F>(TH3Names[n3Histo], TH3Titles[n3Histo],
                                 (int)TH3Binning[n3Histo][0],
                                 TH3Binning[n3Histo][1],
                                 TH3Binning[n3Histo][2],
                                 (int)TH3Binning[n3Histo][3],
                                 TH3Binning[n3Histo][4],
                                 TH3Binning[n3Histo][5],
                                 (int)TH3Binning[n3Histo][6],
                                 TH3Binning[n3Histo][7],
                                 TH3Binning[n3Histo][8]);
      h->GetXaxis()->SetTitle(TH3XaxisTitles[n3Histo]);
      h->GetYaxis()->SetTitle(TH3YaxisTitles[n3Histo]);
      h->GetZaxis()->SetTitle(TH3ZaxisTitles[n3Histo]);
      ++n3Histo;
    }
  }
}

//__________________________________________________________
void GloFwdAssessment::runBasicQC(o2::framework::ProcessingContext& ctx)
{

  // get tracks
  mMFTTracks = ctx.inputs().get<gsl::span<o2::mft::TrackMFT>>("mfttracks");
  mMCHTracks = ctx.inputs().get<gsl::span<o2::mch::TrackMCH>>("mchtracks");
  mGlobalFwdTracks = ctx.inputs().get<gsl::span<o2::dataformats::GlobalFwdTrack>>("fwdtracks");

  if (mUseMC) {
    // get labels
    mMFTTrackLabels = ctx.inputs().get<gsl::span<MCCompLabel>>("mfttrklabels");
    mMCHTrackLabels = ctx.inputs().get<gsl::span<MCCompLabel>>("mchtrklabels");
    mFwdTrackLabels = ctx.inputs().get<gsl::span<MCCompLabel>>("fwdtrklabels");
  }

  for (auto& oneTrack : mGlobalFwdTracks) {
    if (mMIDFilterEnabled and (oneTrack.getMIDMatchingChi2() < 0)) { // MID filter
      continue;
    }
    const auto nClusters = mMFTTracks[oneTrack.getMFTTrackID()].getNumberOfPoints();
    mTrackNumberOfClusters->Fill(nClusters);
    mTrackInvQPt->Fill(oneTrack.getInvQPt());
    mTrackChi2->Fill(oneTrack.getTrackChi2());
    mTrackCharge->Fill(oneTrack.getCharge());
    mTrackPhi->Fill(oneTrack.getPhi());
    mTrackEta->Fill(oneTrack.getEta());
    mTrackCotl->Fill(1. / oneTrack.getTanl());

    for (auto minNClusters : sMinNClustersList) {
      if (nClusters >= minNClusters) {
        mTrackEtaNCls[minNClusters - sMinNClustersList[0]]->Fill(oneTrack.getEta());
        mTrackPhiNCls[minNClusters - sMinNClustersList[0]]->Fill(oneTrack.getPhi());
        mTrackXYNCls[minNClusters - sMinNClustersList[0]]->Fill(oneTrack.getX(), oneTrack.getY());
        mTrackEtaPhiNCls[minNClusters - sMinNClustersList[0]]->Fill(oneTrack.getEta(), oneTrack.getPhi());
      }
    }
  }
}

//__________________________________________________________
void GloFwdAssessment::processGeneratedTracks()
{
  for (auto src = 0; src < mcReader.getNSources(); src++) {
    for (Int_t event = 0; event < mcReader.getNEvents(src); event++) {
      const auto& mcTracks = mcReader.getTracks(src, event);
      auto evh = mcReader.getMCEventHeader(src, event);
      for (const auto& mcParticle : mcTracks) {
        addMCParticletoHistos(&mcParticle, kGen, evh);
      } // mcTracks
      mcReader.releaseTracksForSourceAndEvent(src, event);
    } // events
  }   // sources
}

//__________________________________________________________
void GloFwdAssessment::processPairables()
{
  int trackID = 0, evnID = 0, srcID = 0;
  bool fake = false;
  std::unordered_map<o2::MCCompLabel, std::array<bool, 2>> mcPairables;

  // Loop MCH Tracks
  auto nMCHTracks = mMCHTracks.size();
  for (int iTrk = 0; iTrk < nMCHTracks; ++iTrk) {
    auto mchLabel = mMCHTrackLabels[iTrk];

    mchLabel.get(trackID, evnID, srcID, fake);
    if (fake) {
      continue;
    }
    mcPairables[mchLabel][0] = true;
  }

  // Loop MFT Tracks
  auto nMFTTracks = mMFTTracks.size();
  for (int iTrk = 0; iTrk < nMFTTracks; ++iTrk) {
    auto mftLabel = mMFTTrackLabels[iTrk];

    mftLabel.get(trackID, evnID, srcID, fake);
    if (fake) {
      continue;
    }
    auto t = mcPairables.find(mftLabel);
    if (t != mcPairables.end()) {
      t->second[1] = true;
    }
  }

  // Identify pairables
  mPairableTracksMap.resize(mcReader.getNSources());
  auto src = 0;
  for (auto& map : mPairableTracksMap) {
    map.resize(mcReader.getNEvents(src++));
  }
  for (auto& testPair : mcPairables) {
    auto& boolPair = testPair.second;
    if (boolPair[0] and boolPair[1]) {
      mPairables[testPair.first] = true;
      mPairableTracksMap[testPair.first.getSourceID()][testPair.first.getEventID()].push_back(testPair.first);
    }
  }

  // Process pairables per source and event
  for (auto src = 0; src < mcReader.getNSources(); src++) {
    for (Int_t event = 0; event < mcReader.getNEvents(src); event++) {
      auto evH = mcReader.getMCEventHeader(src, event);
      for (auto& pairable : mPairableTracksMap[src][event]) {
        auto const* mcParticle = mcReader.getTrack(pairable);
        addMCParticletoHistos(mcParticle, kPairable, evH);
      }
      mcReader.releaseTracksForSourceAndEvent(src, event);
    } // events
  }   // sources
}
//__________________________________________________________
void GloFwdAssessment::processRecoTracks()
{
  // For this moment this is used for MC-based assessment, but could be merged into runBasicQC(...)
  for (auto fwdTrack : mGlobalFwdTracks) {
    if (mMIDFilterEnabled and (fwdTrack.getMIDMatchingChi2() < 0)) { // MID filter
      continue;
    }
    auto pt_Rec = fwdTrack.getPt();
    auto invQPt_Rec = fwdTrack.getInvQPt();
    auto px_mch = mMCHTracks[fwdTrack.getMCHTrackID()].getPx();
    auto py_mch = mMCHTracks[fwdTrack.getMCHTrackID()].getPy();
    auto invQPt_MCH = mMCHTracks[fwdTrack.getMCHTrackID()].getSign() / sqrt(px_mch * px_mch + py_mch * py_mch);
    auto eta_Rec = std::abs(fwdTrack.getEta());
    auto phi_Rec = fwdTrack.getPhi();
    auto nMFTClusters = mMFTTracks[fwdTrack.getMFTTrackID()].getNumberOfPoints();
    auto Chi2_Rec = fwdTrack.getTrackChi2();
    int Q_Rec = fwdTrack.getCharge();
    auto matchChi2 = fwdTrack.getMFTMCHMatchingChi2();
    auto matchMatchScore = fwdTrack.getMFTMCHMatchingScore();

    mHistPtVsEta[kReco]->Fill(eta_Rec, pt_Rec);
    mHistPhiVsEta[kReco]->Fill(eta_Rec, phi_Rec);
    mHistPhiVsPt[kReco]->Fill(pt_Rec, phi_Rec);

    mTH3Histos[kTH3GMTrackPtEtaChi2]->Fill(pt_Rec, eta_Rec, matchChi2);
    mTH3Histos[kTH3GMTrackPtEtaMatchScore]->Fill(pt_Rec, eta_Rec, matchMatchScore);
    if (fwdTrack.isCloseMatch()) {
      mTH3Histos[kTH3GMCloseMatchPtEtaChi2]->Fill(pt_Rec, eta_Rec, matchChi2);
      mTH3Histos[kTH3GMCloseMatchPtEtaMatchScore]->Fill(pt_Rec, eta_Rec, matchMatchScore);
    }
  }
}

//__________________________________________________________
void GloFwdAssessment::processTrueTracks()
{
  fillTrueRecoTracksMap();
  for (auto src = 0; src < mcReader.getNSources(); src++) {
    for (Int_t event = 0; event < mcReader.getNEvents(src); event++) {
      const auto& evH = mcReader.getMCEventHeader(src, event);
      const auto& zVtx = evH.GetZ();

      for (const auto& trueFwdTrackID : mTrueTracksMap[src][event]) {
        auto fwdTrack = mGlobalFwdTracks[trueFwdTrackID];
        if (mMIDFilterEnabled and (fwdTrack.getMIDMatchingChi2() < 0)) { // MID filter
          continue;
        }

        const auto& trackLabel = mFwdTrackLabels[trueFwdTrackID];
        if (trackLabel.isCorrect()) {

          auto const* mcParticle = mcReader.getTrack(trackLabel);
          const auto etaGen = std::abs(mcParticle->GetEta());
          const auto phiGen = TMath::ATan2(mcParticle->Py(), mcParticle->Px());
          const auto& ptGen = mcParticle->GetPt();
          const auto& vxGen = mcParticle->GetStartVertexCoordinatesX();
          const auto& vyGen = mcParticle->GetStartVertexCoordinatesY();
          const auto& vzGen = mcParticle->GetStartVertexCoordinatesZ();
          auto tanlGen = mcParticle->Pz() / mcParticle->GetPt();

          const auto& pdgcode_MC = mcParticle->GetPdgCode();
          int Q_Gen;
          if (TDatabasePDG::Instance()->GetParticle(pdgcode_MC)) {
            Q_Gen = TDatabasePDG::Instance()->GetParticle(pdgcode_MC)->Charge() / 3;
          } else {
            continue;
          }
          const auto invQPtGen = 1.0 * Q_Gen / ptGen;
          fwdTrack.propagateToZ(vzGen, mBz);

          const auto& pt_Rec = fwdTrack.getPt();
          const auto& invQPt_Rec = fwdTrack.getInvQPt();
          const auto& px_mch = mMCHTracks[fwdTrack.getMCHTrackID()].getPx();
          const auto& py_mch = mMCHTracks[fwdTrack.getMCHTrackID()].getPy();
          const auto& invQPt_MCH = mMCHTracks[fwdTrack.getMCHTrackID()].getSign() / sqrt(px_mch * px_mch + py_mch * py_mch);
          const auto& eta_Rec = std::abs(fwdTrack.getEta());
          const auto& phi_Rec = fwdTrack.getPhi();
          const auto& nMFTClusters = mMFTTracks[fwdTrack.getMFTTrackID()].getNumberOfPoints();
          const auto& Chi2_Rec = fwdTrack.getTrackChi2();
          const int Q_Rec = fwdTrack.getCharge();
          const auto& matchChi2 = fwdTrack.getMFTMCHMatchingChi2();
          const auto& matchMatchScore = fwdTrack.getMFTMCHMatchingScore();

          // Residuals at vertex
          const auto x_res = fwdTrack.getX() - vxGen;
          const auto y_res = fwdTrack.getY() - vyGen;
          const auto eta_res = fwdTrack.getEta() - etaGen;
          const auto phi_res = fwdTrack.getPhi() - phiGen;
          const auto cotl_res = (1. / fwdTrack.getTanl()) - (1. / tanlGen);
          const auto invQPt_res = invQPt_Rec - invQPtGen;
          mHistPtVsEta[kRecoTrue]->Fill(eta_Rec, pt_Rec);
          mHistPhiVsEta[kRecoTrue]->Fill(eta_Rec, phi_Rec);
          mHistPhiVsPt[kRecoTrue]->Fill(pt_Rec, phi_Rec);
          mHistZvtxVsEta[kRecoTrue]->Fill(eta_Rec, zVtx);

          mHistPhiRecVsPhiGen->Fill(phiGen, phi_Rec);
          mHistEtaRecVsEtaGen->Fill(etaGen, eta_Rec);

          /// Reco assessment histos
          auto d_Charge = Q_Rec - Q_Gen;
          mChargeMatchEff->Fill(!d_Charge, ptGen);

          mTH3Histos[kTH3GMTrackDeltaXVertexPtEta]->Fill(ptGen, etaGen, 1e4 * x_res);
          mTH3Histos[kTH3GMTrackDeltaYVertexPtEta]->Fill(ptGen, etaGen, 1e4 * y_res);
          mTH3Histos[kTH3GMTrackDeltaXDeltaYEta]->Fill(etaGen, 1e4 * x_res, 1e4 * y_res);
          mTH3Histos[kTH3GMTrackDeltaXDeltaYPt]->Fill(ptGen, 1e4 * x_res, 1e4 * y_res);
          mTH3Histos[kTH3GMTrackXPullPtEta]->Fill(ptGen, etaGen, x_res / sqrt(fwdTrack.getCovariances()(0, 0)));
          mTH3Histos[kTH3GMTrackYPullPtEta]->Fill(ptGen, etaGen, y_res / sqrt(fwdTrack.getCovariances()(1, 1)));
          mTH3Histos[kTH3GMTrackPhiPullPtEta]->Fill(ptGen, etaGen, phi_res / sqrt(fwdTrack.getCovariances()(2, 2)));
          mTH3Histos[kTH3GMTrackCotlPullPtEta]->Fill(ptGen, etaGen, cotl_res / sqrt(1. / fwdTrack.getCovariances()(3, 3)));
          mTH3Histos[kTH3GMTrackInvQPtPullPtEta]->Fill(ptGen, etaGen, invQPt_res / sqrt(fwdTrack.getCovariances()(4, 4)));
          mTH3Histos[kTH3GMTrackInvQPtResolutionPtEta]->Fill(ptGen, etaGen, (invQPt_Rec - invQPtGen) / invQPtGen);
          mTH3Histos[kTH3GMTrackInvQPtResMCHPtEta]->Fill(ptGen, etaGen, (invQPt_MCH - invQPtGen) / invQPtGen);
          mTH3Histos[kTH3GMTrackReducedChi2PtEta]->Fill(ptGen, etaGen, Chi2_Rec / (2 * nMFTClusters - 5));
          mTH3Histos[kTH3GMTruePtEtaChi2]->Fill(pt_Rec, eta_Rec, matchChi2);
          mTH3Histos[kTH3GMTruePtEtaMatchScore]->Fill(pt_Rec, eta_Rec, matchMatchScore);
          mTH3Histos[kTH3GMTruePtEtaMatchScore_MC]->Fill(ptGen, etaGen, matchMatchScore);
        }
      }
      mcReader.releaseTracksForSourceAndEvent(src, event);
    } // events
  }   // sources
}

//__________________________________________________________
void GloFwdAssessment::addMCParticletoHistos(const MCTrack* mcTr, const int TrackType, const o2::dataformats::MCEventHeader& evH)
{
  auto zVtx = evH.GetZ();

  auto pt = mcTr->GetPt();
  auto eta = -1 * mcTr->GetEta();
  auto phi = mcTr->GetPhi();
  o2::math_utils::bringToPMPiGend(phi);
  auto z = mcTr->GetStartVertexCoordinatesZ();
  auto R = sqrt(pow(mcTr->GetStartVertexCoordinatesX(), 2) + pow(mcTr->GetStartVertexCoordinatesY(), 2));

  mHistPtVsEta[TrackType]->Fill(eta, pt);
  mHistPhiVsEta[TrackType]->Fill(eta, phi);
  mHistPhiVsPt[TrackType]->Fill(pt, phi);
  mHistZvtxVsEta[TrackType]->Fill(eta, zVtx);
  if (TrackType == kGen || TrackType == kPairable) {
    mHistRVsZ[TrackType]->Fill(z, R);
  }
  if (TrackType == kPairable) {
    mTH3Histos[kTH3GMPairablePtEtaZ]->Fill(pt, eta, zVtx);
  }
}

//__________________________________________________________
void GloFwdAssessment::getHistos(TObjArray& objar)
{

  objar.Add(mTrackNumberOfClusters.get());
  objar.Add(mTrackInvQPt.get());
  objar.Add(mTrackChi2.get());
  objar.Add(mTrackCharge.get());
  objar.Add(mTrackPhi.get());
  objar.Add(mTrackEta.get());
  for (auto minNClusters : sMinNClustersList) {
    auto nHisto = minNClusters - sMinNClustersList[0];
    objar.Add(mTrackEtaNCls[nHisto].get());
    objar.Add(mTrackPhiNCls[nHisto].get());
    objar.Add(mTrackXYNCls[nHisto].get());
    objar.Add(mTrackEtaPhiNCls[nHisto].get());
  }
  objar.Add(mTrackCotl.get());

  if (mUseMC) {
    objar.Add(mHistPhiRecVsPhiGen.get());
    objar.Add(mHistEtaRecVsEtaGen.get());
    for (int TrackType = 0; TrackType < kNumberOfTrackTypes; TrackType++) {
      objar.Add(mHistPhiVsEta[TrackType].get());
      objar.Add(mHistPtVsEta[TrackType].get());
      objar.Add(mHistPhiVsPt[TrackType].get());
      objar.Add(mHistZvtxVsEta[TrackType].get());
      if (TrackType == kGen || TrackType == kPairable) {
        objar.Add(mHistRVsZ[TrackType].get());
      }
    }

    // Histos for Reconstruction assessment

    for (auto& h : mTH3Histos) {
      objar.Add(h.get());
    }

    objar.Add(mChargeMatchEff.get());
    objar.Add(mPairingEtaPt.get());
    objar.Add(mTruePairingEtaPt.get());

    if (mFinalizeAnalysis) {
      objar.Add(mHistVxtOffsetProjection.get());
    }

    if (mFinalizeAnalysis) {
      for (int slicedCanvas = 0; slicedCanvas < kNSlicedTH3; slicedCanvas++) {
        objar.Add(mSlicedCanvas[slicedCanvas]);
      }
      for (int matchingCanvas = 0; matchingCanvas < kNGMAssesmentCanvases; matchingCanvas++) {
        objar.Add(mAssessmentCanvas[matchingCanvas]);
      }
    }
  }
}

//__________________________________________________________
void GloFwdAssessment::loadHistos()
{
  if (mFinalizeAnalysis) {
    throw std::runtime_error("MFTAssessment error: data already loaded");
  }
  mFinalizeAnalysis = true;

  TObjArray* objar;

  TFile* f = new TFile(Form("GlobalForwardAssessment.root"));

  mTrackNumberOfClusters = std::unique_ptr<TH1F>((TH1F*)f->Get("mGlobalFwdNumberOfMFTClusters"));

  mTrackInvQPt = std::unique_ptr<TH1F>((TH1F*)f->Get("mGlobalFwdInvQPt"));

  mTrackChi2 = std::unique_ptr<TH1F>((TH1F*)f->Get("mGlobalFwdChi2"));

  mTrackCharge = std::unique_ptr<TH1F>((TH1F*)f->Get("mGlobalFwdCharge"));

  mTrackPhi = std::unique_ptr<TH1F>((TH1F*)f->Get("mGlobalFwdPhi"));

  mTrackEta = std::unique_ptr<TH1F>((TH1F*)f->Get("mGlobalFwdEta"));

  for (auto minNClusters : sMinNClustersList) {
    auto nHisto = minNClusters - sMinNClustersList[0];
    mTrackEtaNCls[nHisto] = std::unique_ptr<TH1F>((TH1F*)f->Get(Form("mGlobalFwdEta_%d_MinClusters", minNClusters)));

    mTrackPhiNCls[nHisto] = std::unique_ptr<TH1F>((TH1F*)f->Get(Form("mGlobalFwdPhi_%d_MinClusters", minNClusters)));

    mTrackXYNCls[nHisto] = std::unique_ptr<TH2F>((TH2F*)f->Get(Form("mGlobalFwdXY_%d_MinClusters", minNClusters)));

    mTrackEtaPhiNCls[nHisto] = std::unique_ptr<TH2F>((TH2F*)f->Get(Form("mGlobalFwdEtaPhi_%d_MinClusters", minNClusters)));
  }

  mTrackCotl = std::unique_ptr<TH1F>((TH1F*)f->Get("mGlobalFwdCotl"));

  // Creating MC-based histos
  if (mUseMC) {

    mHistPhiRecVsPhiGen = std::unique_ptr<TH2F>((TH2F*)f->Get("mGMTrackPhiRecVsPhiGen"));

    mHistEtaRecVsEtaGen = std::unique_ptr<TH2F>((TH2F*)f->Get("mGMTrackEtaRecVsEtaGen"));

    for (int trackType = 0; trackType < kNumberOfTrackTypes; trackType++) {
      mHistPhiVsEta[trackType] = std::unique_ptr<TH2F>((TH2F*)f->Get((std::string("mGMTrackPhiVsEta") + mNameOfTrackTypes[trackType]).c_str()));

      mHistPtVsEta[trackType] = std::unique_ptr<TH2F>((TH2F*)f->Get((std::string("mGMTrackPtVsEta") + mNameOfTrackTypes[trackType]).c_str()));

      mHistPhiVsPt[trackType] = std::unique_ptr<TH2F>((TH2F*)f->Get((std::string("mGMTrackPhiVsPt") + mNameOfTrackTypes[trackType]).c_str()));

      if (trackType != kReco) {
        mHistZvtxVsEta[trackType] = std::unique_ptr<TH2F>((TH2F*)f->Get((std::string("mGMTrackZvtxVsEta") + mNameOfTrackTypes[trackType]).c_str()));
      }
      if (trackType == kGen || trackType == kPairable) {
        mHistRVsZ[trackType] = std::unique_ptr<TH2F>((TH2F*)f->Get((std::string("mGMTrackRVsZ") + mNameOfTrackTypes[trackType]).c_str()));
      }
    }

    // Histos for Reconstruction assessment
    mChargeMatchEff = std::unique_ptr<TEfficiency>((TEfficiency*)f->Get("mGMTrackQMatchEff"));

    const int nTH3Histos = TH3Names.size();
    auto n3Histo = 0;
    for (auto& h : mTH3Histos) {
      h = std::unique_ptr<TH3F>((TH3F*)f->Get(TH3Names[n3Histo]));
      ++n3Histo;
    }
  }
}

//__________________________________________________________
void GloFwdAssessment::finalizeAnalysis()
{

  mHistVxtOffsetProjection = std::unique_ptr<TH2F>((TH2F*)mTH3Histos[kTH3GMTrackDeltaXDeltaYEta]->Project3D("colz yz"));
  mHistVxtOffsetProjection->SetNameTitle("Vertex_XY_OffSet", "Vertex Offset");
  mHistVxtOffsetProjection->SetOption("COLZ");

  finalizeRecoAndPairables();
  finalizePurityAndEff();
}

//__________________________________________________________
void GloFwdAssessment::finalizeRecoAndPairables()
{
  if (mFinalizeAnalysis) {
    std::vector<float> ptList({.5, 5., 10., 18.0});
    float ptWindow = 1.0;
    std::vector<float> etaList({2.5, 3.0});
    float etaWindow = 0.5;

    std::vector<float> sliceList;
    float sliceWindow;

    for (int nCanvas = 0; nCanvas < kNSlicedTH3; nCanvas++) {
      if (nCanvas % 2) {
        sliceList = etaList;
        sliceWindow = etaWindow;
      } else {
        sliceList = ptList;
        sliceWindow = ptWindow;
      }
      mSlicedCanvas[nCanvas] = new TCanvas(TH3SlicedNames[nCanvas], TH3SlicedNames[nCanvas], 1080, 1080);
      mSlicedCanvas[nCanvas]->UseCurrentStyle();
      mSlicedCanvas[nCanvas]->cd();
      TH3Slicer(mSlicedCanvas[nCanvas], mTH3Histos[TH3SlicedMap[nCanvas]], sliceList, sliceWindow, 2);
    }

    auto& Reco = mTH3Histos[kTH3GMTrackPtEtaMatchScore];
    auto& hTrue = mTH3Histos[kTH3GMTruePtEtaMatchScore];
    auto& hTrue_MC = mTH3Histos[kTH3GMTruePtEtaMatchScore_MC];
    auto& hPairable = mTH3Histos[kTH3GMPairablePtEtaZ];

    auto RecoEtaPt = (TH2D*)Reco->Project3D("xy COLZ");
    auto TrueEtaPt_MC = (TH2D*)hTrue_MC->Project3D("xy COLZ");
    auto PairableEtaPt = (TH2D*)hPairable->Project3D("xy COLZ");

    mPairingEtaPt = (std::unique_ptr<TH2D>)static_cast<TH2D*>(RecoEtaPt->Clone());
    mPairingEtaPt->Divide(PairableEtaPt);
    mPairingEtaPt->SetNameTitle("GMTrackPairingEffEtaPt", "PairingEffEtaPt");
    mPairingEtaPt->SetOption("COLZ");

    mTruePairingEtaPt = (std::unique_ptr<TH2D>)static_cast<TH2D*>(TrueEtaPt_MC->Clone());
    mTruePairingEtaPt->Divide(PairableEtaPt);
    mTruePairingEtaPt->SetNameTitle("GMTrackTruePairingEffEtaPt", "TruePairingEffEtaPt");
    mTruePairingEtaPt->SetOption("COLZ");
  }
}

//__________________________________________________________
void GloFwdAssessment::finalizePurityAndEff()
{

  gStyle->SetOptTitle(kFALSE);
  gStyle->SetOptStat(0); // Remove title of first histogram from canvas
  gStyle->SetMarkerStyle(kFullCircle);
  gStyle->SetMarkerSize(1.0);

  auto& Reco = mTH3Histos[kTH3GMTrackPtEtaMatchScore];
  auto& hTrue = mTH3Histos[kTH3GMTruePtEtaMatchScore];
  auto& hTrue_MC = mTH3Histos[kTH3GMTruePtEtaMatchScore_MC];
  auto& hPairable = mTH3Histos[kTH3GMPairablePtEtaZ];

  // Inner pseudorapidity
  auto minBin = Reco->GetYaxis()->FindBin(2.4);
  auto midBin = Reco->GetYaxis()->FindBin(3.0);
  auto maxBin = Reco->GetYaxis()->FindBin(3.6);
  auto PairablePtProjInner = (TH1*)hPairable->ProjectionX("PairableInner", midBin, maxBin);
  auto PairablePtProjOuter = (TH1*)hPairable->ProjectionX("PairableOuter", minBin, midBin);

  auto RecoEtaPt = (TH2D*)Reco->Project3D("xy COLZ");
  auto TrueEtaPt = (TH2D*)hTrue->Project3D("xy COLZ");
  auto TrueEtaPt_MC = (TH2D*)hTrue_MC->Project3D("xy COLZ");
  auto PairableEtaPt = (TH2D*)hPairable->Project3D("xy COLZ");
  auto PairablePt = (TH1D*)hPairable->Project3D("x");

  /// Purity vs score cuts
  float scoreStep = (mFinalizeMaxCut - mFinalizeMinCut) / mNFinalizeSteps;
  for (float scoreCut = mFinalizeMinCut; scoreCut <= mFinalizeMaxCut; scoreCut += scoreStep) {

    auto RecoPtProj = (TH1*)Reco->ProjectionX(Form("_RecoPtProj%.2f", scoreCut));
    auto TruePtProj = (TH1*)hTrue->ProjectionX(Form("_TruePtProj%.2f", scoreCut));
    auto TruePtProj_MC = (TH1*)hTrue_MC->ProjectionX(Form("_TruePtProj_MC%.2f", scoreCut));

    // Inner pseudorapidity
    auto maxScoreBin = Reco->GetZaxis()->FindBin(scoreCut);
    auto RecoPtProjInner = (TH1*)Reco->ProjectionX(Form("_InnerRecoCut_%.2f", scoreCut), midBin, maxBin, 0, maxScoreBin);
    auto TruePtProjInner = (TH1*)hTrue->ProjectionX(Form("_InnerTrueCut_%.2f", scoreCut), midBin, maxBin, 0, maxScoreBin);
    auto TruePtProjInner_MC = (TH1*)hTrue_MC->ProjectionX(Form("_InnerTrueCut_MC_%.2f", scoreCut), midBin, maxBin, 0, maxScoreBin);

    auto& hPurityInner = mPurityPtInnerVecTH2.emplace_back((std::unique_ptr<TH2D>)static_cast<TH2D*>(TruePtProjInner->Clone()));
    hPurityInner->Divide(RecoPtProjInner); // Global Pairing Purity = N_true / N_reco
    hPurityInner->SetNameTitle(Form("TH2GMTrackPurityInnerEtaCut_%.2f", scoreCut), Form("%.2f cut", scoreCut));
    hPurityInner->GetYaxis()->SetTitle("Pairing Purity [ N_{True} / N_{Rec}]");
    hPurityInner->SetOption("COLZ");
    hPurityInner->SetMarkerStyle(kFullCircle);
    hPurityInner->SetMinimum(0.0);
    hPurityInner->SetMaximum(1.2);

    auto& hPairingEffInner = mPairingPtInnerVecTH1.emplace_back((std::unique_ptr<TH1D>)static_cast<TH1D*>(RecoPtProjInner->Clone()));
    hPairingEffInner->Divide(PairablePtProjInner); // Pairing Efficiency = N_reco / N_Pairable
    hPairingEffInner->SetNameTitle(Form("GMTrackPairingEffInnerPtCut_%.2f", scoreCut), Form("%.2f cut", scoreCut));
    hPairingEffInner->GetYaxis()->SetTitle("Pairing Efficiency [ N_{Rec} / N_{pairable}]");
    hPairingEffInner->SetOption("COLZ");
    hPairingEffInner->SetMarkerStyle(kFullCircle);
    hPairingEffInner->SetMinimum(0.0);
    hPairingEffInner->SetMaximum(1.8);

    auto& hTruePairingEffInner = mTruePairingPtInnerVecTH1.emplace_back((std::unique_ptr<TH1D>)static_cast<TH1D*>(TruePtProjInner_MC->Clone()));
    hTruePairingEffInner->Divide(PairablePtProjInner);
    hTruePairingEffInner->SetNameTitle(Form("GMTrackTruePairingEffInnerPtCut_%.2f", scoreCut), Form("%.2f cut", scoreCut));
    hTruePairingEffInner->GetYaxis()->SetTitle("True Pairing Efficiency [ N_{True} / N_{pairable}]");
    hTruePairingEffInner->SetOption("COLZ");
    hTruePairingEffInner->SetMarkerStyle(kFullCircle);
    hTruePairingEffInner->SetMinimum(0.0);
    hTruePairingEffInner->SetMaximum(1.2);

    // Outer pseudorapidity
    auto RecoPtProjOuter = (TH1*)Reco->ProjectionX(Form("_OuterRecoCut_%.2f", scoreCut), minBin, midBin, 0, maxScoreBin);
    auto TruePtProjOuter = (TH1*)hTrue->ProjectionX(Form("_OuterTrueCut_%.2f", scoreCut), minBin, midBin, 0, maxScoreBin);
    auto TruePtProjOuter_MC = (TH1*)hTrue_MC->ProjectionX(Form("_OuterTrueCut_MC_%.2f", scoreCut), minBin, midBin, 0, maxScoreBin);

    auto& hPurityOuter = mPurityPtOuterVecTH2.emplace_back((std::unique_ptr<TH2D>)static_cast<TH2D*>(TruePtProjOuter->Clone()));
    hPurityOuter->Divide(RecoPtProjOuter); // Global Pairing Purity = N_true / N_reco
    hPurityOuter->SetNameTitle(Form("TH2GMTrackPurityOuterEtaCut_%.2f", scoreCut), Form("%.2f cut", scoreCut));
    hPurityOuter->GetYaxis()->SetTitle("Pairing Purity [ N_{True} / N_{Rec}]");
    hPurityOuter->SetOption("COLZ");
    hPurityOuter->SetMarkerStyle(kFullTriangleUp);
    hPurityOuter->SetMinimum(0.0);
    hPurityOuter->SetMaximum(1.2);

    auto& hPairingEffOuter = mPairingPtOuterVecTH1.emplace_back((std::unique_ptr<TH1D>)static_cast<TH1D*>(RecoPtProjOuter->Clone()));
    hPairingEffOuter->Divide(PairablePtProjInner); // Pairing Efficiency = N_reco / N_Pairable
    hPairingEffOuter->SetNameTitle(Form("GMTrackPairingEffOuterPtCut_%.2f", scoreCut), Form("%.2f cut", scoreCut));
    hPairingEffOuter->GetYaxis()->SetTitle("Pairing Efficiency [ N_{Rec} / N_{pairable}]");
    hPairingEffOuter->SetOption("COLZ");
    hPairingEffOuter->SetMarkerStyle(kFullTriangleUp);
    hPairingEffOuter->SetMinimum(0.0);
    hPairingEffOuter->SetMaximum(1.8);

    auto& hTruePairingEffOuter = mTruePairingPtOuterVecTH1.emplace_back((std::unique_ptr<TH1D>)static_cast<TH1D*>(TruePtProjOuter_MC->Clone()));
    hTruePairingEffOuter->Divide(PairablePtProjOuter);
    hTruePairingEffOuter->SetNameTitle(Form("GMTrackTruePairingEffOuterPtCut_%.2f", scoreCut), Form("%.2f cut", scoreCut));
    hTruePairingEffOuter->GetYaxis()->SetTitle("True Pairing Efficiency [ N_{True} / N_{pairable}]");
    hTruePairingEffOuter->SetOption("COLZ");
    hTruePairingEffOuter->SetMarkerStyle(kFullTriangleUp);
    hTruePairingEffOuter->SetMinimum(0.0);
    hTruePairingEffOuter->SetMaximum(1.2);

    mPairingEtaPtVec.emplace_back((std::unique_ptr<TH2D>)static_cast<TH2D*>(RecoEtaPt->Clone()));
    mPairingEtaPtVec.back()->Divide(PairableEtaPt); // Pairing Efficiency = N_reco / N_Pairable
    mPairingEtaPtVec.back()->SetNameTitle(Form("GMTrackPairingEffEtaPtCut_%.2f", scoreCut), Form("%.2f", scoreCut));
    mPairingEtaPtVec.back()->SetOption("COLZ");

    mTruePairingEtaPtVec.emplace_back((std::unique_ptr<TH2D>)static_cast<TH2D*>(TrueEtaPt_MC->Clone()));
    mTruePairingEtaPtVec.back()->Divide(PairableEtaPt);
    mTruePairingEtaPtVec.back()->SetNameTitle(Form("GMTrackTruePairingEffEtaPtCut_%.2f", scoreCut), Form("%.2f", scoreCut));
    mTruePairingEtaPtVec.back()->SetOption("COLZ");
  }

  auto nCanvas = kPurityPtOuter;
  auto canvasName = GMAssesmentNames[nCanvas];
  mAssessmentCanvas[nCanvas] = new TCanvas(canvasName, canvasName, 1080, 800);
  mAssessmentCanvas[nCanvas]->UseCurrentStyle();
  mAssessmentCanvas[nCanvas]->cd();
  auto first = true;
  std::string option;

  std::vector<float> verylowPtOuterPurity;
  std::vector<float> verylowPtInnerPurity;
  std::vector<float> verylowPtOuterEff;
  std::vector<float> verylowPtInnerEff;
  std::vector<float> veryLowPtOuterTrueEff;
  std::vector<float> veryLowPtInnerTrueEff;
  std::vector<float> lowPtOuterPurity;
  std::vector<float> lowPtInnerPurity;
  std::vector<float> lowPtOuterEff;
  std::vector<float> lowPtInnerEff;
  std::vector<float> lowPtOuterTrueEff;
  std::vector<float> lowPtInnerTrueEff;
  std::vector<float> highPtOuterPurity;
  std::vector<float> highPtInnerPurity;
  std::vector<float> highPtOuterEff;
  std::vector<float> highPtInnerEff;
  std::vector<float> highPtOuterTrueEff;
  std::vector<float> highPtInnerTrueEff;

  auto veryLowptBin = mPurityPtOuterVecTH2.front()->GetXaxis()->FindBin(0.25);
  auto lowptBin = mPurityPtOuterVecTH2.front()->GetXaxis()->FindBin(0.75);
  auto highptBin = mPurityPtOuterVecTH2.front()->GetXaxis()->FindBin(2.25);

  for (auto& th2 : mPurityPtOuterVecTH2) {
    if (first) {
      option = "hist P PMC";
    } else {
      option = "hist SAME P PMC";
    }
    first = false;
    th2->Draw(option.c_str());

    verylowPtOuterPurity.push_back(th2->GetBinContent(veryLowptBin));
    lowPtOuterPurity.push_back(th2->GetBinContent(lowptBin));
    highPtOuterPurity.push_back(th2->GetBinContent(highptBin));
  }
  TPaveText* t = new TPaveText(0.2223748, 0.9069355, 0.7776252, 0.965, "brNDC");
  t->SetBorderSize(0);
  t->SetFillColor(gStyle->GetTitleFillColor());
  t->AddText("Global Muon Track Purity (2.4 < #eta < 3.0)");
  t->Draw();

  mAssessmentCanvas[nCanvas]->BuildLegend(.8, .15, .96, .87);
  mAssessmentCanvas[nCanvas]->SetTicky();
  mAssessmentCanvas[nCanvas]->SetGridy();

  nCanvas = kPurityPtInner;
  canvasName = GMAssesmentNames[nCanvas];
  mAssessmentCanvas[nCanvas] = new TCanvas(canvasName, canvasName, 1080, 800);
  mAssessmentCanvas[nCanvas]->UseCurrentStyle();
  mAssessmentCanvas[nCanvas]->cd();
  first = true;

  for (auto& th2 : mPurityPtInnerVecTH2) {
    if (first) {
      option = "hist P PMC";
    } else {
      option = "hist SAME P PMC";
    }
    first = false;
    th2->Draw(option.c_str());

    verylowPtInnerPurity.push_back(th2->GetBinContent(veryLowptBin));
    lowPtInnerPurity.push_back(th2->GetBinContent(lowptBin));
    highPtInnerPurity.push_back(th2->GetBinContent(highptBin));
  }
  t = new TPaveText(0.2223748, 0.9069355, 0.7776252, 0.965, "brNDC");
  t->SetBorderSize(0);
  t->SetFillColor(gStyle->GetTitleFillColor());
  t->AddText("Global Muon Track Purity (3.0 < #eta < 3.6)");
  t->Draw();

  mAssessmentCanvas[nCanvas]->BuildLegend(.8, .15, .96, .87);
  mAssessmentCanvas[nCanvas]->SetTicky();
  mAssessmentCanvas[nCanvas]->SetGridy();

  nCanvas = kPairingEffPtOuter;
  canvasName = GMAssesmentNames[nCanvas];
  mAssessmentCanvas[nCanvas] = new TCanvas(canvasName, canvasName, 1080, 800);
  mAssessmentCanvas[nCanvas]->UseCurrentStyle();
  mAssessmentCanvas[nCanvas]->cd();
  first = true;

  for (auto& th2 : mPairingPtOuterVecTH1) {
    if (first) {
      option = "hist P PMC";
    } else {
      option = "hist SAME P PMC";
    }
    first = false;
    verylowPtOuterEff.push_back(th2->GetBinContent(veryLowptBin));
    lowPtOuterEff.push_back(th2->GetBinContent(lowptBin));
    highPtOuterEff.push_back(th2->GetBinContent(highptBin));
    th2->Draw(option.c_str());
  }
  t = new TPaveText(0.2223748, 0.9069355, 0.7776252, 0.965, "brNDC");
  t->SetBorderSize(0);
  t->SetFillColor(gStyle->GetTitleFillColor());
  t->AddText("Global Muon Track Pairing Efficiency (2.4 < #eta < 3.0)");
  t->Draw();

  mAssessmentCanvas[nCanvas]->BuildLegend(.8, .15, .96, .87);
  mAssessmentCanvas[nCanvas]->SetTicky();
  mAssessmentCanvas[nCanvas]->SetGridy();

  nCanvas = kPairingEffPtInner;
  canvasName = GMAssesmentNames[nCanvas];
  mAssessmentCanvas[nCanvas] = new TCanvas(canvasName, canvasName, 1080, 800);
  mAssessmentCanvas[nCanvas]->UseCurrentStyle();
  mAssessmentCanvas[nCanvas]->cd();
  first = true;

  for (auto& th2 : mPairingPtInnerVecTH1) {
    if (first) {
      option = "hist P PMC";
    } else {
      option = "hist SAME P PMC";
    }
    first = false;
    verylowPtInnerEff.push_back(th2->GetBinContent(veryLowptBin));
    lowPtInnerEff.push_back(th2->GetBinContent(lowptBin));
    highPtInnerEff.push_back(th2->GetBinContent(highptBin));
    th2->Draw(option.c_str());
  }
  t = new TPaveText(0.2223748, 0.9069355, 0.7776252, 0.965, "brNDC");
  t->SetBorderSize(0);
  t->SetFillColor(gStyle->GetTitleFillColor());
  t->AddText("Global Muon Track Pairing Efficiency (3.0 < #eta < 3.6 )");
  t->Draw();

  mAssessmentCanvas[nCanvas]->BuildLegend(.8, .15, .96, .87);
  mAssessmentCanvas[nCanvas]->SetTicky();
  mAssessmentCanvas[nCanvas]->SetGridy();

  nCanvas = kTruePairingEffPtOuter;
  canvasName = GMAssesmentNames[nCanvas];
  mAssessmentCanvas[nCanvas] = new TCanvas(canvasName, canvasName, 1080, 800);
  mAssessmentCanvas[nCanvas]->UseCurrentStyle();
  mAssessmentCanvas[nCanvas]->cd();
  first = true;

  for (auto& th2 : mTruePairingPtOuterVecTH1) {
    if (first) {
      option = "hist P PMC";
    } else {
      option = "hist SAME P PMC";
    }
    first = false;
    veryLowPtOuterTrueEff.push_back(th2->GetBinContent(veryLowptBin));
    lowPtOuterTrueEff.push_back(th2->GetBinContent(lowptBin));
    highPtOuterTrueEff.push_back(th2->GetBinContent(highptBin));
    th2->Draw(option.c_str());
  }
  t = new TPaveText(0.2223748, 0.9069355, 0.7776252, 0.965, "brNDC");
  t->SetBorderSize(0);
  t->SetFillColor(gStyle->GetTitleFillColor());
  t->AddText("Global Muon Track True Pairing Efficiency (2.4 < #eta < 3.0)");
  t->Draw();

  mAssessmentCanvas[nCanvas]->BuildLegend(.8, .15, .96, .87);
  mAssessmentCanvas[nCanvas]->SetTicky();
  mAssessmentCanvas[nCanvas]->SetGridy();

  nCanvas = kTruePairingEffPtInner;
  canvasName = GMAssesmentNames[nCanvas];
  mAssessmentCanvas[nCanvas] = new TCanvas(canvasName, canvasName, 1080, 800);
  mAssessmentCanvas[nCanvas]->UseCurrentStyle();
  mAssessmentCanvas[nCanvas]->cd();
  first = true;

  for (auto& th2 : mTruePairingPtInnerVecTH1) {
    if (first) {
      option = "hist P PMC";
    } else {
      option = "hist SAME P PMC";
    }
    first = false;
    veryLowPtInnerTrueEff.push_back(th2->GetBinContent(veryLowptBin));
    lowPtInnerTrueEff.push_back(th2->GetBinContent(lowptBin));
    highPtInnerTrueEff.push_back(th2->GetBinContent(highptBin));
    th2->Draw(option.c_str());
  }
  t = new TPaveText(0.2223748, 0.9069355, 0.7776252, 0.965, "brNDC");
  t->SetBorderSize(0);
  t->SetFillColor(gStyle->GetTitleFillColor());
  t->AddText("Global Muon Track True Pairing Efficiency (3.0 < #eta < 3.6 )");
  t->Draw();

  mAssessmentCanvas[nCanvas]->BuildLegend(.8, .15, .96, .87);
  mAssessmentCanvas[nCanvas]->SetTicky();
  mAssessmentCanvas[nCanvas]->SetGridy();

  nCanvas = kPurityVsEfficiency;
  canvasName = GMAssesmentNames[nCanvas];
  mAssessmentCanvas[nCanvas] = new TCanvas(canvasName, canvasName, 1080, 800);

  TGraph* gr = new TGraph(highPtInnerEff.size(), &highPtInnerEff[0], &highPtInnerPurity[0]);
  gr->SetMinimum(0);
  gr->SetMaximum(1.01);

  gr->SetMarkerStyle(kFullCircle);
  gr->Draw("A P PMC");
  gr->GetXaxis()->SetTitle("Global Muon Pairing Efficiency [ N_{Rec} / N_{pairable}]");
  gr->GetXaxis()->SetLimits(0.f, 1.6);
  gr->GetYaxis()->SetTitle("Pairing Purity [ N_{True} / N_{Rec}]");
  gr->SetTitle("p_{t} = 2.25 || (3.0 < #eta < 3.6 )");

  gr = new TGraph(highPtOuterEff.size(), &highPtOuterEff[0], &highPtOuterPurity[0]);
  gr->Draw("P PMC SAME");
  gr->SetMarkerStyle(kFullTriangleUp);

  gr->SetTitle("p_{t} = 2.25 || (2.4 < #eta < 3.0)");

  gr = new TGraph(lowPtInnerEff.size(), &lowPtInnerEff[0], &lowPtInnerPurity[0]);
  gr->Draw("P PMC SAME");
  gr->SetTitle("p_{t} = 0.75 || (3.0 < #eta < 3.6 )");

  gr = new TGraph(lowPtOuterEff.size(), &lowPtOuterEff[0], &lowPtOuterPurity[0]);
  gr->Draw("P PMC SAME");
  gr->SetMarkerStyle(kFullTriangleUp);
  gr->SetTitle("p_{t} = 0.75 || (2.4 < #eta < 3.0)");

  gr = new TGraph(verylowPtInnerEff.size(), &verylowPtInnerEff[0], &verylowPtInnerPurity[0]);
  gr->Draw("P PMC SAME");
  gr->SetTitle("p_{t} = 0.25 || (3.0 < #eta < 3.6)");

  gr = new TGraph(verylowPtOuterEff.size(), &verylowPtOuterEff[0], &verylowPtOuterPurity[0]);
  gr->Draw("P PMC SAME");
  gr->SetMarkerStyle(kFullTriangleUp);

  gr->SetTitle("p_{t} = 0.25 || (2.4 < #eta < 3.0)");

  mAssessmentCanvas[nCanvas]->BuildLegend();
  mAssessmentCanvas[nCanvas]->SetTicky();
  mAssessmentCanvas[nCanvas]->SetGridy();

  nCanvas = kPurityVsTrueEfficiency;
  canvasName = GMAssesmentNames[nCanvas];
  mAssessmentCanvas[nCanvas] = new TCanvas(canvasName, canvasName, 1080, 800);

  TGraph* gr2 = new TGraph(highPtInnerTrueEff.size(), &highPtInnerTrueEff[0], &highPtInnerPurity[0]);
  gr2->SetMinimum(0);
  gr2->SetMaximum(1.01);

  gr2->SetMarkerStyle(kFullCircle);
  gr2->Draw("A P PMC");
  gr2->GetXaxis()->SetTitle("Global Muon True Pairing Efficiency [ N_{True} / N_{pairable}]");
  gr2->GetXaxis()->SetLimits(0.f, 1.01);
  gr2->GetYaxis()->SetTitle("Pairing Purity [ N_{True} / N_{Rec}]");
  gr2->SetTitle("p_{t} = 2.25 || (3.0 < #eta < 3.6 )");

  gr2 = new TGraph(highPtOuterTrueEff.size(), &highPtOuterTrueEff[0], &highPtOuterPurity[0]);
  gr2->Draw("P PMC SAME");
  gr2->SetMarkerStyle(kFullTriangleUp);

  gr2->SetTitle("p_{t} = 2.25 || (2.4 < #eta < 3.0)");

  gr2 = new TGraph(lowPtInnerTrueEff.size(), &lowPtInnerTrueEff[0], &lowPtInnerPurity[0]);
  gr2->Draw("P PMC SAME");
  gr2->SetTitle("p_{t} = 0.75 || (3.0 < #eta < 3.6 )");

  gr2 = new TGraph(lowPtOuterTrueEff.size(), &lowPtOuterTrueEff[0], &lowPtOuterPurity[0]);
  gr2->Draw("P PMC SAME");
  gr2->SetMarkerStyle(kFullTriangleUp);
  gr2->SetTitle("p_{t} = 0.75 || (2.4 < #eta < 3.0)");

  gr2 = new TGraph(veryLowPtInnerTrueEff.size(), &veryLowPtInnerTrueEff[0], &verylowPtInnerPurity[0]);
  gr2->Draw("P PMC SAME");
  gr2->SetTitle("p_{t} = 0.25 || (3.0 < #eta < 3.6)");

  gr2 = new TGraph(veryLowPtOuterTrueEff.size(), &veryLowPtOuterTrueEff[0], &verylowPtOuterPurity[0]);
  gr2->Draw("P PMC SAME");
  gr2->SetMarkerStyle(kFullTriangleUp);

  gr2->SetTitle("p_{t} = 0.25 || (2.4 < #eta < 3.0)");

  mAssessmentCanvas[nCanvas]->BuildLegend();
  mAssessmentCanvas[nCanvas]->SetTicky();
  mAssessmentCanvas[nCanvas]->SetGridy();
}

//__________________________________________________________
void GloFwdAssessment::TH3Slicer(TCanvas* canvas, std::unique_ptr<TH3F>& histo3D, std::vector<float> list, double window, int iPar, float marker_size)
{
  gStyle->SetOptTitle(kFALSE);
  gStyle->SetOptStat(0); // Remove title of first histogram from canvas
  gStyle->SetMarkerStyle(kFullCircle);
  gStyle->SetMarkerSize(marker_size);
  canvas->UseCurrentStyle();
  canvas->cd();
  std::string cname = canvas->GetName();
  std::string ctitle = cname;
  std::string option;
  std::string option2 = "PLC PMC same";

  TObjArray aSlices;
  histo3D->GetYaxis()->SetRange(0, 0);
  histo3D->GetXaxis()->SetRange(0, 0);
  bool first = true;
  if (cname.find("VsEta") < cname.length()) {
    for (auto ptmin : list) {
      auto ptmax = ptmin + window;
      histo3D->GetXaxis()->SetRangeUser(ptmin, ptmax);

      std::string ytitle = "\\sigma (";
      ytitle += histo3D->GetZaxis()->GetTitle();
      ytitle += ")";
      auto title = Form("_%1.2f_%1.2f_yz", ptmin, ptmax);
      auto aDBG = (TH2F*)histo3D->Project3D(title);
      aDBG->GetXaxis()->SetRangeUser(0, 0);

      aDBG->FitSlicesX(nullptr, 0, -1, 4, "QNR", &aSlices);
      auto th1DBG = (TH1F*)aSlices[iPar];
      th1DBG->SetTitle(Form("%1.2f < p_t < %1.2f", ptmin, ptmax));
      th1DBG->SetStats(0);
      th1DBG->SetYTitle(ytitle.c_str());
      if (first) {
        option = "PLC PMC";
      } else {
        option = "SAME PLC PMC";
      }
      first = false;
      th1DBG->DrawClone(option.c_str());
    }
  } else if (cname.find("VsPt") < cname.length()) {
    for (auto etamin : list) {
      auto etamax = etamin + window;
      histo3D->GetYaxis()->SetRangeUser(etamin, etamax);
      std::string ytitle = "\\sigma (" + std::string(histo3D->GetZaxis()->GetTitle()) + ")";
      auto title = Form("_%1.2f_%1.2f_xz", etamin, etamax);
      auto aDBG = (TH2F*)histo3D->Project3D(title);
      aDBG->FitSlicesX(nullptr, 0, -1, 4, "QNR", &aSlices);
      auto th1DBG = (TH1F*)aSlices[iPar];
      th1DBG->SetTitle(Form("%1.2f < \\eta < %1.2f", etamin, etamax));
      th1DBG->SetStats(0);
      th1DBG->SetYTitle(ytitle.c_str());
      if (first) {
        option = "PLC PMC";
      } else {
        option = "SAME PLC PMC";
      }
      first = false;
      th1DBG->DrawClone(option.c_str());
    }
  } else {
    exit(1);
  }

  histo3D->GetYaxis()->SetRange(0, 0);
  histo3D->GetXaxis()->SetRange(0, 0);

  TPaveText* t = new TPaveText(0.2223748, 0.9069355, 0.7776252, 0.965, "brNDC"); // left-up
  t->SetBorderSize(0);
  t->SetFillColor(gStyle->GetTitleFillColor());
  t->AddText(ctitle.c_str());
  t->Draw();

  canvas->BuildLegend();
  canvas->SetTicky();
  canvas->SetGridy();
  if (0) {
    cname += ".png";
    canvas->Print(cname.c_str());
  }
}
