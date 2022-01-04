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

  mTrackTanl->Reset();

  if (mUseMC) {
    mPairables.clear();
    mMFTTrackables.clear();

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

    auto h = mChargeMatchEff->GetCopyTotalHisto();
    h->Reset();
    mChargeMatchEff->SetTotalHistogram(*h, "");
    mChargeMatchEff->SetPassedHistogram(*h, "");

    for (auto& h : mTH3Histos) {
      h->Reset();
    }
  }
}

//__________________________________________________________
void GloFwdAssessment::createHistos()
{

  // Creating data-only histos
  mTrackNumberOfClusters = std::make_unique<TH1F>("mGlobalFwdNumberOfClusters",
                                                  "Number Of Clusters Per Track; # clusters; # entries", 10, 0.5, 10.5);

  mTrackInvQPt = std::make_unique<TH1F>("mGlobalFwdInvQPt", "Track q/p_{T}; q/p_{T} [1/GeV]; # entries", 50, -2, 2);

  mTrackChi2 = std::make_unique<TH1F>("mGlobalFwdChi2", "Track #chi^{2}; #chi^{2}; # entries", 21, -0.5, 20.5);

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

  mTrackTanl = std::make_unique<TH1F>("mGlobalFwdTanl", "Track tan #lambda; tan #lambda; # entries", 100, -25, 0);

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
    mTrackTanl->Fill(oneTrack.getTanl());

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
void GloFwdAssessment::processPairables()
{
  int trackID = 0, evnID = 0, srcID = 0;
  bool fake = false;
  std::unordered_map<o2::MCCompLabel, std::array<bool, 2>> mcPairables;

  // Loop MCH Tracks
  auto nMCHTracks = mMCHTracks.size();
  for (int iTrk = 0; iTrk < nMCHTracks; ++iTrk) {
    auto mchLabel = mMCHTrackLabels[iTrk];
    // std::cout << "Starting pairability mchLabel = " << mchLabel << std::endl;

    mchLabel.get(trackID, evnID, srcID, fake);
    if (fake) {
      continue;
    }
    // mcPairables.insert_or_assign(mchLabel, std::array<bool, 2>({true, false}));
    // mcPairables.insert
    mcPairables[mchLabel][0] = true;
  }

  // Loop MFT Tracks
  auto nMFTTracks = mMFTTracks.size();
  for (int iTrk = 0; iTrk < nMFTTracks; ++iTrk) {
    auto mftLabel = mMFTTrackLabels[iTrk];
    // std::cout << " Testing pairability mftLabel = " << mftLabel << std::endl;

    mftLabel.get(trackID, evnID, srcID, fake);
    if (fake) {
      continue;
    }
    auto t = mcPairables.find(mftLabel);
    if (t != mcPairables.end()) {
      t->second[1] = true;
      // mcPairables[mftLabel][1] = true;
    }
  }

  // Loop MFT Tracks

  // Identify and process pairables
  for (auto& testPair : mcPairables) {
    auto& boolPair = testPair.second;
    if (boolPair[0] and boolPair[1]) {
      mPairables[testPair.first] = true;
      auto const* mcParticle = mcReader.getTrack(testPair.first);
      srcID = testPair.first.getSourceID();
      evnID = testPair.first.getEventID();
      auto evH = mcReader.getMCEventHeader(srcID, evnID);
      addMCParticletoHistos(mcParticle, kPairable, evH);
    }
  }
}

//__________________________________________________________
void GloFwdAssessment::processRecoAndTrueTracks()
{

  auto trkId = 0;
  for (auto fwdTrack : mGlobalFwdTracks) {
    if (mMIDFilterEnabled and (fwdTrack.getMIDMatchingChi2() < 0)) { // MID filter
      trkId++;
      continue;
    }
    auto pt_Rec = fwdTrack.getPt();
    auto invQPt_Rec = fwdTrack.getInvQPt();
    auto eta_Rec = std::abs(fwdTrack.getEta());
    auto phi_Rec = fwdTrack.getPhi();
    auto nMFTClusters = mMFTTracks[fwdTrack.getMFTTrackID()].getNumberOfPoints();
    auto Chi2_Rec = fwdTrack.getTrackChi2();
    int Q_Rec = fwdTrack.getCharge();

    auto TrackType = kReco;
    mHistPtVsEta[TrackType]->Fill(eta_Rec, pt_Rec);
    mHistPhiVsEta[TrackType]->Fill(eta_Rec, phi_Rec);
    mHistPhiVsPt[TrackType]->Fill(pt_Rec, phi_Rec);

    auto trackLabel = mFwdTrackLabels[trkId];
    if (trackLabel.isCorrect()) {
      TrackType = kRecoTrue;
      auto evH = mcReader.getMCEventHeader(trackLabel.getSourceID(), trackLabel.getEventID());
      auto zVtx = evH.GetZ();

      auto const* mcParticle = mcReader.getTrack(trackLabel);
      auto etaGen = std::abs(mcParticle->GetEta());
      auto phiGen = TMath::ATan2(mcParticle->Py(), mcParticle->Px());
      auto ptGen = mcParticle->GetPt();
      auto vxGen = mcParticle->GetStartVertexCoordinatesX();
      auto vyGen = mcParticle->GetStartVertexCoordinatesY();
      auto vzGen = mcParticle->GetStartVertexCoordinatesZ();
      auto tanlGen = mcParticle->Pz() / mcParticle->GetPt();

      auto pdgcode_MC = mcParticle->GetPdgCode();
      int Q_Gen;
      if (TDatabasePDG::Instance()->GetParticle(pdgcode_MC)) {
        Q_Gen = TDatabasePDG::Instance()->GetParticle(pdgcode_MC)->Charge() / 3;
      } else {
        continue;
      }
      auto invQPtGen = 1.0 * Q_Gen / ptGen;
      fwdTrack.propagateToZ(vzGen, mBz);

      // Residuals at vertex
      auto x_res = fwdTrack.getX() - vxGen;
      auto y_res = fwdTrack.getY() - vyGen;
      auto eta_res = fwdTrack.getEta() - etaGen;
      auto phi_res = fwdTrack.getPhi() - phiGen;
      auto tanl_res = fwdTrack.getTanl() - tanlGen;
      auto invQPt_res = invQPt_Rec - invQPtGen;
      mHistPtVsEta[TrackType]->Fill(eta_Rec, pt_Rec);
      mHistPhiVsEta[TrackType]->Fill(eta_Rec, phi_Rec);
      mHistPhiVsPt[TrackType]->Fill(pt_Rec, phi_Rec);
      mHistZvtxVsEta[TrackType]->Fill(eta_Rec, zVtx);

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
      mTH3Histos[kTH3GMTrackTanlPullPtEta]->Fill(ptGen, etaGen, tanl_res / sqrt(fwdTrack.getCovariances()(3, 3)));
      mTH3Histos[kTH3GMTrackInvQPtPullPtEta]->Fill(ptGen, etaGen, invQPt_res / sqrt(fwdTrack.getCovariances()(4, 4)));
      mTH3Histos[kTH3GMTrackInvQPtResolutionPtEta]->Fill(ptGen, etaGen, (invQPt_Rec - invQPtGen) / invQPtGen);
      mTH3Histos[kTH3GMTrackReducedChi2PtEta]->Fill(ptGen, etaGen, Chi2_Rec / (2 * nMFTClusters - 5)); // 5: number of fitting parameters
    }
    trkId++;
  }
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
  objar.Add(mTrackTanl.get());

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
  }
}

//__________________________________________________________
bool GloFwdAssessment::loadHistos()
{

  return true;
}

//__________________________________________________________
void GloFwdAssessment::finalizeAnalysis()
{
}