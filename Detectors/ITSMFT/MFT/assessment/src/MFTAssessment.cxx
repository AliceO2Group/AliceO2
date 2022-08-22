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

#include "MFTAssessment/MFTAssessment.h"
#include "Framework/InputSpec.h"
#include "MFTBase/GeometryTGeo.h"
#include "MathUtils/Utils.h"
#include <Framework/InputRecord.h>
#include <TPaveText.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TFile.h>

using namespace o2::mft;
using o2::itsmft::CompClusterExt;
o2::itsmft::ChipMappingMFT mMFTMapping;

//__________________________________________________________
void MFTAssessment::init(bool finalizeAnalysis)
{
  mFinalizeAnalysis = finalizeAnalysis;
  createHistos();
  mUnusedChips.fill(true);
}

//__________________________________________________________
void MFTAssessment::reset()
{

  mTrackNumberOfClusters->Reset();
  mCATrackNumberOfClusters->Reset();
  mLTFTrackNumberOfClusters->Reset();
  mTrackOnvQPt->Reset();
  mTrackChi2->Reset();
  mTrackCharge->Reset();
  mTrackPhi->Reset();
  mPositiveTrackPhi->Reset();
  mNegativeTrackPhi->Reset();
  mTrackEta->Reset();

  mMFTClsZ->Reset();
  mMFTClsOfTracksZ->Reset();

  mUnusedChips.fill(true);
  mNumberTFs = 0;

  for (auto nMFTLayer = 0; nMFTLayer < 10; nMFTLayer++) {
    mMFTClsXYinLayer[nMFTLayer]->Reset();
    mMFTClsOfTracksXYinLayer[nMFTLayer]->Reset();
  }

  for (auto nMFTDisk = 0; nMFTDisk < 5; nMFTDisk++) {
    mMFTClsXYRedundantInDisk[nMFTDisk]->Reset();
  }

  for (auto minNClusters : sMinNClustersList) {
    auto nHisto = minNClusters - sMinNClustersList[0];
    mTrackEtaNCls[nHisto]->Reset();
    mTrackPhiNCls[nHisto]->Reset();
    mTrackXYNCls[nHisto]->Reset();
    mTrackEtaPhiNCls[nHisto]->Reset();
  }
  mCATrackEta->Reset();
  mLTFTrackEta->Reset();
  mTrackTanl->Reset();

  mTrackROFNEntries->Reset();
  mClusterROFNEntries->Reset();

  mClusterSensorIndex->Reset();
  mClusterPatternIndex->Reset();

  if (mUseMC) {
    mMFTTrackables.clear();
    mTrueTracksMap.clear();
    mTrackableTracksMap.clear();

    mHistPhiRecVsPhiGen->Reset();
    mHistEtaRecVsEtaGen->Reset();
    for (int trackType = 0; trackType < kNumberOfTrackTypes; trackType++) {
      mHistPhiVsEta[trackType]->Reset();
      mHistPtVsEta[trackType]->Reset();
      mHistPhiVsPt[trackType]->Reset();
      mHistZvtxVsEta[trackType]->Reset();
      if (trackType == kGen || trackType == kTrackable) {
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
void MFTAssessment::createHistos()
{
  auto MaxClusterROFSize = 5000;
  auto MaxTrackROFSize = 1000;
  auto ROFLengthInBC = 198;
  auto ROFsPerOrbit = o2::constants::lhc::LHCMaxBunches / ROFLengthInBC;
  auto MaxDuration = 60.f;
  auto TimeBinSize = 0.01f;
  auto NofTimeBins = static_cast<int>(MaxDuration / TimeBinSize);

  // Creating data-only histos
  mTrackNumberOfClusters = std::make_unique<TH1F>("mMFTTrackNumberOfClusters",
                                                  "Number Of Clusters Per Track; # clusters; # entries", 10, 0.5, 10.5);

  mCATrackNumberOfClusters = std::make_unique<TH1F>("mMFTCATrackNumberOfClusters",
                                                    "Number Of Clusters Per CA Track; # clusters; # tracks", 10, 0.5, 10.5);

  mLTFTrackNumberOfClusters = std::make_unique<TH1F>("mMFTLTFTrackNumberOfClusters",
                                                     "Number Of Clusters Per LTF Track; # clusters; # entries", 10, 0.5, 10.5);

  mTrackOnvQPt = std::make_unique<TH1F>("mMFTTrackOnvQPt", "Track q/p_{T}; q/p_{T} [1/GeV]; # entries", 50, -2, 2);

  mTrackChi2 = std::make_unique<TH1F>("mMFTTrackChi2", "Track #chi^{2}; #chi^{2}; # entries", 21, -0.5, 20.5);

  mTrackCharge = std::make_unique<TH1F>("mMFTTrackCharge", "Track Charge; q; # entries", 3, -1.5, 1.5);

  mTrackPhi = std::make_unique<TH1F>("mMFTTrackPhi", "Track #phi; #phi; # entries", 100, -3.2, 3.2);

  mPositiveTrackPhi = std::make_unique<TH1F>("mMFTPositiveTrackPhi", "Positive Track #phi; #phi; # entries", 100, -3.2, 3.2);

  mNegativeTrackPhi = std::make_unique<TH1F>("mMFTNegativeTrackPhi", "Negative Track #phi; #phi; # entries", 100, -3.2, 3.2);

  mTrackEta = std::make_unique<TH1F>("mMFTTrackEta", "Track #eta; #eta; # entries", 50, -4, -2);

  //----------------------------------------------------------------------------

  mMFTClsZ = std::make_unique<TH1F>("mMFTClsZ", "Z of all clusters; Z (cm); # entries", 400, -80, -40);

  mMFTClsOfTracksZ = std::make_unique<TH1F>("mMFTClsOfTracksZ", "Z of clusters belonging to MFT tracks; Z (cm); # entries", 400, -80, -40);

  for (auto nMFTLayer = 0; nMFTLayer < 10; nMFTLayer++) {
    mMFTClsXYinLayer[nMFTLayer] = std::make_unique<TH2F>(Form("mMFTClsXYinLayer%d", nMFTLayer), Form("Cluster Position in Layer %d; x (cm); y (cm)", nMFTLayer), 400, -20, 20, 400, -20, 20);
    mMFTClsOfTracksXYinLayer[nMFTLayer] = std::make_unique<TH2F>(Form("mMFTClsOfTracksXYinLayer%d", nMFTLayer), Form("Cluster (of MFT tracks) Position in Layer %d; x (cm); y (cm)", nMFTLayer), 400, -20, 20, 400, -20, 20);
  }

  for (auto nMFTDisk = 0; nMFTDisk < 5; nMFTDisk++) {
    mMFTClsXYRedundantInDisk[nMFTDisk] = std::make_unique<TH2F>(Form("mMFTClsXYRedundantInDisk%d", nMFTDisk), Form("Redondant Cluster Position in disk %d; x (cm); y (cm)", nMFTDisk), 400, -20, 20, 400, -20, 20);
  }

  for (auto minNClusters : sMinNClustersList) {
    auto nHisto = minNClusters - sMinNClustersList[0];
    mTrackEtaNCls[nHisto] = std::make_unique<TH1F>(Form("mMFTTrackEta_%d_MinClusters", minNClusters), Form("Track #eta (NCls >= %d); #eta; # entries", minNClusters), 50, -4, -2);

    mTrackPhiNCls[nHisto] = std::make_unique<TH1F>(Form("mMFTTrackPhi_%d_MinClusters", minNClusters), Form("Track #phi (NCls >= %d); #phi; # entries", minNClusters), 100, -3.2, 3.2);

    mTrackXYNCls[nHisto] = std::make_unique<TH2F>(Form("mMFTTrackXY_%d_MinClusters", minNClusters), Form("Track Position (NCls >= %d); x; y", minNClusters), 320, -16, 16, 320, -16, 16);
    mTrackXYNCls[nHisto]->SetOption("COLZ");

    mTrackEtaPhiNCls[nHisto] = std::make_unique<TH2F>(Form("mMFTTrackEtaPhi_%d_MinClusters", minNClusters), Form("Track #eta , #phi (NCls >= %d); #eta; #phi", minNClusters), 50, -4, -2, 100, -3.2, 3.2);
    mTrackEtaPhiNCls[nHisto]->SetOption("COLZ");
  }

  mCATrackEta = std::make_unique<TH1F>("mMFTCATrackEta", "CA Track #eta; #eta; # entries", 50, -4, -2);

  mLTFTrackEta = std::make_unique<TH1F>("mMFTLTFTrackEta", "LTF Track #eta; #eta; # entries", 50, -4, -2);

  mTrackTanl = std::make_unique<TH1F>("mMFTTrackTanl", "Track tan #lambda; tan #lambda; # entries", 100, -25, 0);

  mClusterROFNEntries = std::make_unique<TH1F>("mMFTClustersROFSize", "MFT Cluster ROFs size; ROF Size; # entries", MaxClusterROFSize, 0, MaxClusterROFSize);

  mTrackROFNEntries = std::make_unique<TH1F>("mMFTTrackROFSize", "MFT Track ROFs size; ROF Size; # entries", MaxTrackROFSize, 0, MaxTrackROFSize);

  mTracksBC = std::make_unique<TH1F>("mMFTTracksBC", "Tracks per BC (sum over orbits); BCid; # entries", ROFsPerOrbit, 0, o2::constants::lhc::LHCMaxBunches);
  mTracksBC->SetMinimum(0.1);

  mNOfTracksTime = std::make_unique<TH1F>("mNOfTracksTime", "Number of tracks per time bin; time (s); # entries", NofTimeBins, 0, MaxDuration);
  mNOfTracksTime->SetMinimum(0.1);

  mNOfClustersTime = std::make_unique<TH1F>("mNOfClustersTime", "Number of clusters per time bin; time (s); # entries", NofTimeBins, 0, MaxDuration);
  mNOfClustersTime->SetMinimum(0.1);

  mClusterSensorIndex = std::make_unique<TH1F>("mMFTClusterSensorIndex", "Chip Cluster Occupancy;Chip ID;#Entries", 936, -0.5, 935.5);

  mClusterPatternIndex = std::make_unique<TH1F>("mMFTClusterPatternIndex", "Cluster Pattern ID;Pattern ID;#Entries", 300, -0.5, 299.5);

  // Creating MC-based histos
  if (mUseMC) {

    LOG(info) << "Initializing MC Reader";
    if (!mMCReader.isInitialized()) {
      if (!mMCReader.initFromDigitContext("collisioncontext.root")) {
        throw std::invalid_argument("initialization of MCKinematicsReader failed");
      }
    }

    mHistPhiRecVsPhiGen = std::make_unique<TH2F>("mHistPhiRecVsPhiGen", "Phi Rec Vs Phi Gen of true reco tracks ", 24, 0, 2 * TMath::Pi(), 24, 0, 2 * TMath::Pi());
    mHistPhiRecVsPhiGen->SetXTitle((std::string("#phi of ") + mNameOfTrackTypes[kGen]).c_str());
    mHistPhiRecVsPhiGen->SetYTitle((std::string("#phi of ") + mNameOfTrackTypes[kRecoTrue]).c_str());
    mHistPhiRecVsPhiGen->Sumw2();
    mHistPhiRecVsPhiGen->SetOption("COLZ");

    mHistEtaRecVsEtaGen = std::make_unique<TH2F>("mHistEtaRecVsEtaGen", "Eta Rec Vs Eta Gen of true reco tracks ", 35, -4.5, -1.0, 35, -4.5, -1.0);
    mHistEtaRecVsEtaGen->SetXTitle((std::string("#eta of ") + mNameOfTrackTypes[kGen]).c_str());
    mHistEtaRecVsEtaGen->SetYTitle((std::string("#eta of ") + mNameOfTrackTypes[kRecoTrue]).c_str());
    mHistEtaRecVsEtaGen->Sumw2();
    mHistEtaRecVsEtaGen->SetOption("COLZ");

    for (int trackType = 0; trackType < kNumberOfTrackTypes; trackType++) {
      // mHistPhiVsEta
      mHistPhiVsEta[trackType] = std::make_unique<TH2F>((std::string("mHistPhiVsEta") + mNameOfTrackTypes[trackType]).c_str(), (std::string("Phi Vs Eta of ") + mNameOfTrackTypes[trackType]).c_str(), 35, -4.5, -1, 24, 0, 2 * TMath::Pi());
      mHistPhiVsEta[trackType]->SetXTitle((std::string("#eta of ") + mNameOfTrackTypes[trackType]).c_str());
      mHistPhiVsEta[trackType]->SetYTitle((std::string("#phi of ") + mNameOfTrackTypes[trackType]).c_str());
      mHistPhiVsEta[trackType]->Sumw2();
      mHistPhiVsEta[trackType]->SetOption("COLZ");

      // mHistPtVsEta
      mHistPtVsEta[trackType] = std::make_unique<TH2F>((std::string("mHistPtVsEta") + mNameOfTrackTypes[trackType]).c_str(), (std::string("Pt Vs Eta of ") + mNameOfTrackTypes[trackType]).c_str(), 35, -4.5, -1, 40, 0., 10.);
      mHistPtVsEta[trackType]->SetXTitle((std::string("#eta of ") + mNameOfTrackTypes[trackType]).c_str());
      mHistPtVsEta[trackType]->SetYTitle((std::string("p_{T} (GeV/c) of ") + mNameOfTrackTypes[trackType]).c_str());
      mHistPtVsEta[trackType]->Sumw2();
      mHistPtVsEta[trackType]->SetOption("COLZ");

      // mHistPhiVsPt
      mHistPhiVsPt[trackType] = std::make_unique<TH2F>((std::string("mHistPhiVsPt") + mNameOfTrackTypes[trackType]).c_str(), (std::string("Phi Vs Pt of ") + mNameOfTrackTypes[trackType]).c_str(), 40, 0., 10., 24, 0, 2 * TMath::Pi());
      mHistPhiVsPt[trackType]->SetXTitle((std::string("p_{T} (GeV/c) of ") + mNameOfTrackTypes[trackType]).c_str());
      mHistPhiVsPt[trackType]->SetYTitle((std::string("#phi of ") + mNameOfTrackTypes[trackType]).c_str());
      mHistPhiVsPt[trackType]->Sumw2();
      mHistPhiVsPt[trackType]->SetOption("COLZ");

      if (trackType != kReco) {
        // mHistZvtxVsEta
        mHistZvtxVsEta[trackType] = std::make_unique<TH2F>((std::string("mHistZvtxVsEta") + mNameOfTrackTypes[trackType]).c_str(), (std::string("Z_{vtx} Vs Eta of ") + mNameOfTrackTypes[trackType]).c_str(), 35, -4.5, -1, 15, -15, 15);
        mHistZvtxVsEta[trackType]->SetXTitle((std::string("#eta of ") + mNameOfTrackTypes[trackType]).c_str());
        mHistZvtxVsEta[trackType]->SetYTitle((std::string("z_{vtx} (cm) of ") + mNameOfTrackTypes[trackType]).c_str());
        mHistZvtxVsEta[trackType]->Sumw2();
        mHistZvtxVsEta[trackType]->SetOption("COLZ");
      }
      // mHistRVsZ]
      if (trackType == kGen || trackType == kTrackable) {
        mHistRVsZ[trackType] = std::make_unique<TH2F>((std::string("mHistRVsZ") + mNameOfTrackTypes[trackType]).c_str(), (std::string("R Vs Z of ") + mNameOfTrackTypes[trackType]).c_str(), 400, -80., 20., 400, 0., 80.);
        mHistRVsZ[trackType]->SetXTitle((std::string("z (cm) origin of ") + mNameOfTrackTypes[trackType]).c_str());
        mHistRVsZ[trackType]->SetYTitle((std::string("R (cm) radius of origin of ") + mNameOfTrackTypes[trackType]).c_str());
        mHistRVsZ[trackType]->Sumw2();
        mHistRVsZ[trackType]->SetOption("COLZ");
      }
    }

    // Histos for Reconstruction assessment

    mChargeMatchEff = std::make_unique<TEfficiency>("QMatchEff", "Charge Match;p_t [GeV];#epsilon", 50, 0, 20);

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
void MFTAssessment::runASyncQC(o2::framework::ProcessingContext& ctx)
{
  mNumberTFs++; // TF Counter

  // get tracks
  mMFTTracks = ctx.inputs().get<gsl::span<o2::mft::TrackMFT>>("tracks");
  mMFTTracksROF = ctx.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("tracksrofs");
  mMFTTrackClusIdx = ctx.inputs().get<gsl::span<int>>("trackClIdx");

  // get clusters
  mMFTClusters = ctx.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("compClusters");
  mMFTClustersROF = ctx.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("clustersrofs");
  mMFTClusterPatterns = ctx.inputs().get<gsl::span<unsigned char>>("patterns");
  pattIt = mMFTClusterPatterns.begin();
  mMFTClustersGlobal.clear();
  mMFTClustersGlobal.reserve(mMFTClusters.size());
  o2::mft::ioutils::convertCompactClusters(mMFTClusters, pattIt, mMFTClustersGlobal, mDictionary);

  if (mUseMC) {
    // get labels
    mMFTClusterLabels = ctx.inputs().get<const dataformats::MCTruthContainer<MCCompLabel>*>("clslabels");
    mMFTTrackLabels = ctx.inputs().get<gsl::span<MCCompLabel>>("trklabels");
  }

  // Fill the clusters histograms
  for (const auto& rof : mMFTClustersROF) {
    mClusterROFNEntries->Fill(rof.getNEntries());
    float seconds = orbitToSeconds(rof.getBCData().orbit, mRefOrbit) + rof.getBCData().bc * o2::constants::lhc::LHCBunchSpacingNS * 1e-9;
    mNOfClustersTime->Fill(seconds, rof.getNEntries());
  }

  for (int icls = 0; icls < mMFTClusters.size(); ++icls) {
    auto const oneCluster = mMFTClusters[icls];
    auto const globalCluster = mMFTClustersGlobal[icls];

    mClusterSensorIndex->Fill(oneCluster.getSensorID());
    mClusterPatternIndex->Fill(oneCluster.getPatternID());

    mMFTClsZ->Fill(globalCluster.getZ());

    auto clsLayer = mMFTChipMapper.chip2Layer(oneCluster.getChipID());
    mMFTClsXYinLayer[clsLayer]->Fill(globalCluster.getX(), globalCluster.getY());

    mUnusedChips[oneCluster.getChipID()] = false; // this chipID is used
  }

  // fill the tracks histogram

  for (const auto& rof : mMFTTracksROF) {
    mTrackROFNEntries->Fill(rof.getNEntries());
    mTracksBC->Fill(rof.getBCData().bc, rof.getNEntries());
    float seconds = orbitToSeconds(rof.getBCData().orbit, mRefOrbit) + rof.getBCData().bc * o2::constants::lhc::LHCBunchSpacingNS * 1e-9;
    mNOfTracksTime->Fill(seconds, rof.getNEntries());
  }

  std::array<std::array<int, 2>, 5> clsEntriesForRedundancy;

  for (auto& oneTrack : mMFTTracks) {
    mTrackNumberOfClusters->Fill(oneTrack.getNumberOfPoints());
    mTrackChi2->Fill(oneTrack.getTrackChi2());
    mTrackCharge->Fill(oneTrack.getCharge());
    mTrackPhi->Fill(oneTrack.getPhi());
    mTrackEta->Fill(oneTrack.getEta());
    mTrackTanl->Fill(oneTrack.getTanl());

    for (auto idisk = 0; idisk < 5; idisk++) {
      clsEntriesForRedundancy[idisk] = {-1, -1};
    }

    auto ncls = oneTrack.getNumberOfPoints();
    auto offset = oneTrack.getExternalClusterIndexOffset();

    for (int icls = 0; icls < ncls; ++icls) // cluster loop
    {

      auto clsEntry = mMFTTrackClusIdx[offset + icls];
      auto globalCluster = mMFTClustersGlobal[clsEntry];

      mMFTClsOfTracksZ->Fill(globalCluster.getZ());

      auto layer = mMFTMapping.ChipID2Layer[globalCluster.getSensorID()];

      mMFTClsOfTracksXYinLayer[layer]->Fill(globalCluster.getX(), globalCluster.getY());

      int clsMFTdiskID = layer / 2;

      if (clsEntriesForRedundancy[clsMFTdiskID][0] != -1) {
        clsEntriesForRedundancy[clsMFTdiskID][1] = clsEntry;
      } else {
        clsEntriesForRedundancy[clsMFTdiskID][0] = clsEntry;
      }
    }

    for (auto idisk = 0; idisk < 5; idisk++) {
      if ((clsEntriesForRedundancy[idisk][0] != -1) && (clsEntriesForRedundancy[idisk][1] != -1)) {
        auto globalCluster1 = mMFTClustersGlobal[clsEntriesForRedundancy[idisk][0]];

        mMFTClsXYRedundantInDisk[idisk]->Fill(globalCluster1.getX(), globalCluster1.getY());
      }
    }

    for (auto minNClusters : sMinNClustersList) {
      if (oneTrack.getNumberOfPoints() >= minNClusters) {
        mTrackEtaNCls[minNClusters - sMinNClustersList[0]]->Fill(oneTrack.getEta());
        mTrackPhiNCls[minNClusters - sMinNClustersList[0]]->Fill(oneTrack.getPhi());
        mTrackXYNCls[minNClusters - sMinNClustersList[0]]->Fill(oneTrack.getX(), oneTrack.getY());
        mTrackEtaPhiNCls[minNClusters - sMinNClustersList[0]]->Fill(oneTrack.getEta(), oneTrack.getPhi());
      }
    }

    if (oneTrack.getCharge() == +1) {
      mPositiveTrackPhi->Fill(oneTrack.getPhi());
      mTrackOnvQPt->Fill(1 / oneTrack.getPt());
    }

    if (oneTrack.getCharge() == -1) {
      mNegativeTrackPhi->Fill(oneTrack.getPhi());
      mTrackOnvQPt->Fill(-1 / oneTrack.getPt());
    }

    if (oneTrack.isCA()) {
      mCATrackNumberOfClusters->Fill(oneTrack.getNumberOfPoints());
      mCATrackEta->Fill(oneTrack.getEta());
    }
    if (oneTrack.isLTF()) {
      mLTFTrackNumberOfClusters->Fill(oneTrack.getNumberOfPoints());
      mLTFTrackEta->Fill(oneTrack.getEta());
    }
  }
}

//__________________________________________________________
void MFTAssessment::processGeneratedTracks()
{
  for (auto src = 0; src < mMCReader.getNSources(); src++) {
    for (Int_t event = 0; event < mMCReader.getNEvents(src); event++) {
      auto evh = mMCReader.getMCEventHeader(src, event);
      const auto& mcTracks = mMCReader.getTracks(src, event);
      for (const auto& mcParticle : mcTracks) {
        addMCParticletoHistos(&mcParticle, kGen, evh);
      } // mcTracks
      mMCReader.releaseTracksForSourceAndEvent(src, event);
    } // events
  }   // sources
}

//__________________________________________________________
void MFTAssessment::processTrackables()
{
  int trackID = 0, evnID = 0, srcID = 0;
  bool fake = false;

  std::unordered_map<o2::MCCompLabel, std::array<bool, 5>> mcTrackHasClusterInMFTDisks;

  auto nClusters = mMFTClusters.size();
  for (int icls = 0; icls < nClusters; ++icls) {
    auto const cluster = mMFTClusters[icls];
    auto labelSize = (mMFTClusterLabels->getLabels(icls)).size();
    for (auto il = 0; il < labelSize; il++) {
      auto clsLabel = (mMFTClusterLabels->getLabels(icls))[il];
      clsLabel.get(trackID, evnID, srcID, fake);
      if (fake) {
        continue;
      }
      auto clsLayer = mMFTChipMapper.chip2Layer(cluster.getChipID());
      int clsMFTdiskID = clsLayer / 2;
      mcTrackHasClusterInMFTDisks[clsLabel][clsMFTdiskID] = true;
    }
  } // loop on clusters

  // Identify trackables
  mTrackableTracksMap.resize(mMCReader.getNSources());
  auto src = 0;
  for (auto& map : mTrackableTracksMap) {
    map.resize(mMCReader.getNEvents(src++));
  }

  for (auto& trackClsInDisk : mcTrackHasClusterInMFTDisks) {
    auto& clsdisk = trackClsInDisk.second;
    auto nMFTDisks = 0;
    for (auto disk : {0, 1, 2, 3, 4}) {
      nMFTDisks += int(clsdisk[disk]);
    }
    if (nMFTDisks >= 4) {
      mMFTTrackables[trackClsInDisk.first] = true;
      mTrackableTracksMap[trackClsInDisk.first.getSourceID()][trackClsInDisk.first.getEventID()].push_back(trackClsInDisk.first);
    }
  }

  // Process trackables
  for (auto src = 0; src < mMCReader.getNSources(); src++) {
    for (Int_t event = 0; event < mMCReader.getNEvents(src); event++) {
      auto evH = mMCReader.getMCEventHeader(src, event);
      for (auto& trackable : mTrackableTracksMap[src][event]) {
        auto const* mcParticle = mMCReader.getTrack(trackable);
        addMCParticletoHistos(mcParticle, kTrackable, evH);
      }
      mMCReader.releaseTracksForSourceAndEvent(src, event);
    } // events
  }   // sources
}

//__________________________________________________________
void MFTAssessment::addMCParticletoHistos(const MCTrack* mcTr, const int TrackType, const o2::dataformats::MCEventHeader& evH)
{
  auto zVtx = evH.GetZ();

  auto pt = mcTr->GetPt();
  auto eta = mcTr->GetEta();
  float phi = TMath::ATan2(mcTr->Py(), mcTr->Px());
  o2::math_utils::bringTo02Pi(phi);
  auto z = mcTr->GetStartVertexCoordinatesZ();
  auto R = sqrt(pow(mcTr->GetStartVertexCoordinatesX(), 2) + pow(mcTr->GetStartVertexCoordinatesY(), 2));

  mHistPtVsEta[TrackType]->Fill(eta, pt);
  mHistPhiVsEta[TrackType]->Fill(eta, phi);
  mHistPhiVsPt[TrackType]->Fill(pt, phi);
  mHistZvtxVsEta[TrackType]->Fill(eta, zVtx);
  if (TrackType == kGen || TrackType == kTrackable) {
    mHistRVsZ[TrackType]->Fill(z, R);
  }
}

//__________________________________________________________
void MFTAssessment::processRecoTracks()
{
  // For this moment this is used for MC-based assessment, but could be merged into runASyncQC(...)
  for (auto mftTrack : mMFTTracks) {
    const auto& pt_Rec = mftTrack.getPt();
    const auto& invQPt_Rec = mftTrack.getInvQPt();
    const auto& invQPt_Seed = mftTrack.getInvQPtSeed();
    const auto& eta_Rec = mftTrack.getEta();
    float phi_Rec = mftTrack.getPhi();
    o2::math_utils::bringTo02Pi(phi_Rec);
    const auto& nClusters = mftTrack.getNumberOfPoints();
    const auto& Chi2_Rec = mftTrack.getTrackChi2();
    int Q_Rec = mftTrack.getCharge();

    mHistPtVsEta[kReco]->Fill(eta_Rec, pt_Rec);
    mHistPhiVsEta[kReco]->Fill(eta_Rec, phi_Rec);
    mHistPhiVsPt[kReco]->Fill(pt_Rec, phi_Rec);
  }
}

//__________________________________________________________
void MFTAssessment::processTrueTracks()
{
  fillTrueRecoTracksMap();
  for (auto src = 0; src < mMCReader.getNSources(); src++) {
    for (Int_t event = 0; event < mMCReader.getNEvents(src); event++) {
      auto evH = mMCReader.getMCEventHeader(src, event);
      auto zVtx = evH.GetZ();

      for (const auto& trueMFTTrackID : mTrueTracksMap[src][event]) {
        auto mftTrack = mMFTTracks[trueMFTTrackID];
        const auto& trackLabel = mMFTTrackLabels[trueMFTTrackID];
        if (trackLabel.isCorrect()) {
          auto const* mcParticle = mMCReader.getTrack(trackLabel);
          auto pdgcode_MC = mcParticle->GetPdgCode();
          int Q_Gen;
          if (TDatabasePDG::Instance()->GetParticle(pdgcode_MC)) {
            Q_Gen = TDatabasePDG::Instance()->GetParticle(pdgcode_MC)->Charge() / 3;
          } else {
            continue;
          }

          auto etaGen = mcParticle->GetEta();
          float phiGen = TMath::ATan2(mcParticle->Py(), mcParticle->Px());
          o2::math_utils::bringTo02Pi(phiGen);
          auto ptGen = mcParticle->GetPt();
          auto vxGen = mcParticle->GetStartVertexCoordinatesX();
          auto vyGen = mcParticle->GetStartVertexCoordinatesY();
          auto vzGen = mcParticle->GetStartVertexCoordinatesZ();
          auto tanlGen = mcParticle->Pz() / mcParticle->GetPt();
          auto invQPtGen = 1.0 * Q_Gen / ptGen;

          mftTrack.propagateToZ(vzGen, mBz);
          const auto& pt_Rec = mftTrack.getPt();
          const auto& invQPt_Rec = mftTrack.getInvQPt();
          const auto& invQPt_Seed = mftTrack.getInvQPtSeed();
          const auto& eta_Rec = mftTrack.getEta();
          float phi_Rec = mftTrack.getPhi();
          o2::math_utils::bringTo02Pi(phi_Rec);
          const auto& nClusters = mftTrack.getNumberOfPoints();
          const auto& Chi2_Rec = mftTrack.getTrackChi2();
          int Q_Rec = mftTrack.getCharge();
          // Residuals at vertex
          auto x_res = mftTrack.getX() - vxGen;
          auto y_res = mftTrack.getY() - vyGen;
          auto eta_res = mftTrack.getEta() - etaGen;
          auto phi_res = mftTrack.getPhi() - phiGen;
          auto tanl_res = mftTrack.getTanl() - tanlGen;
          auto invQPt_res = invQPt_Rec - invQPtGen;
          mHistPtVsEta[kRecoTrue]->Fill(eta_Rec, pt_Rec);
          mHistPhiVsEta[kRecoTrue]->Fill(eta_Rec, phi_Rec);
          mHistPhiVsPt[kRecoTrue]->Fill(pt_Rec, phi_Rec);
          mHistZvtxVsEta[kRecoTrue]->Fill(eta_Rec, zVtx);

          mHistPtVsEta[kRecoTrueMC]->Fill(etaGen, ptGen);
          mHistPhiVsEta[kRecoTrueMC]->Fill(etaGen, phiGen);
          mHistPhiVsPt[kRecoTrueMC]->Fill(ptGen, phiGen);
          mHistZvtxVsEta[kRecoTrueMC]->Fill(eta_Rec, zVtx);

          mHistPhiRecVsPhiGen->Fill(phiGen, phi_Rec);
          mHistEtaRecVsEtaGen->Fill(etaGen, eta_Rec);
          /// Reco assessment histos
          auto d_Charge = Q_Rec - Q_Gen;
          mChargeMatchEff->Fill(!d_Charge, ptGen);

          mTH3Histos[kTH3TrackDeltaXVertexPtEta]->Fill(ptGen, etaGen, 1e4 * x_res);
          mTH3Histos[kTH3TrackDeltaYVertexPtEta]->Fill(ptGen, etaGen, 1e4 * y_res);
          mTH3Histos[kTH3TrackDeltaXDeltaYEta]->Fill(etaGen, 1e4 * x_res, 1e4 * y_res);
          mTH3Histos[kTH3TrackDeltaXDeltaYPt]->Fill(ptGen, 1e4 * x_res, 1e4 * y_res);
          mTH3Histos[kTH3TrackXPullPtEta]->Fill(ptGen, etaGen, x_res / sqrt(mftTrack.getCovariances()(0, 0)));
          mTH3Histos[kTH3TrackYPullPtEta]->Fill(ptGen, etaGen, y_res / sqrt(mftTrack.getCovariances()(1, 1)));
          mTH3Histos[kTH3TrackPhiPullPtEta]->Fill(ptGen, etaGen, phi_res / sqrt(mftTrack.getCovariances()(2, 2)));
          mTH3Histos[kTH3TrackTanlPullPtEta]->Fill(ptGen, etaGen, tanl_res / sqrt(mftTrack.getCovariances()(3, 3)));
          mTH3Histos[kTH3TrackInvQPtPullPtEta]->Fill(ptGen, etaGen, invQPt_res / sqrt(mftTrack.getCovariances()(4, 4)));
          mTH3Histos[kTH3TrackInvQPtResolutionPtEta]->Fill(ptGen, etaGen, (invQPt_Rec - invQPtGen) / invQPtGen);
          mTH3Histos[kTH3TrackInvQPtResSeedPtEta]->Fill(ptGen, etaGen, (invQPt_Seed - invQPtGen) / invQPtGen);
          mTH3Histos[kTH3TrackReducedChi2PtEta]->Fill(ptGen, etaGen, Chi2_Rec / (2 * nClusters - 5)); // 5: number of fitting parameters
        }
      }
      mMCReader.releaseTracksForSourceAndEvent(src, event);
    } // events
  }   // sources
}

//__________________________________________________________
void MFTAssessment::getHistos(TObjArray& objar)
{
  TH1F* mMFTDeadChipID = new TH1F("mMFTDeadChipID", "chipID of the dead chips; chipID; # entries", 936, -0.5, 935.5);
  TH1F* mTFsCounter = new TH1F("mTFsCounter", "counter of TFs; count bin; # entries", 3, 0, 2);

  auto chipID = 0;
  for (auto chipState : mUnusedChips) {
    mMFTDeadChipID->Fill(chipID, float(chipState));
    chipID++;
  }
  mTFsCounter->Fill(1, mNumberTFs);

  mMFTDeadChipID->Scale(mNumberTFs);

  objar.Add(mTrackNumberOfClusters.get());
  objar.Add(mCATrackNumberOfClusters.get());
  objar.Add(mLTFTrackNumberOfClusters.get());
  objar.Add(mTrackOnvQPt.get());
  objar.Add(mTrackChi2.get());
  objar.Add(mTrackCharge.get());
  objar.Add(mTrackPhi.get());
  objar.Add(mPositiveTrackPhi.get());
  objar.Add(mNegativeTrackPhi.get());
  objar.Add(mTrackEta.get());

  //------
  objar.Add(mMFTClsZ.get());
  objar.Add(mMFTClsOfTracksZ.get());
  objar.Add(mMFTDeadChipID);
  objar.Add(mTFsCounter);
  for (auto nMFTLayer = 0; nMFTLayer < 10; nMFTLayer++) {
    objar.Add(mMFTClsXYinLayer[nMFTLayer].get());
    objar.Add(mMFTClsOfTracksXYinLayer[nMFTLayer].get());
  }

  for (auto nMFTDisk = 0; nMFTDisk < 5; nMFTDisk++) {
    objar.Add(mMFTClsXYRedundantInDisk[nMFTDisk].get());
  }

  for (auto minNClusters : sMinNClustersList) {
    auto nHisto = minNClusters - sMinNClustersList[0];
    objar.Add(mTrackEtaNCls[nHisto].get());
    objar.Add(mTrackPhiNCls[nHisto].get());
    objar.Add(mTrackXYNCls[nHisto].get());
    objar.Add(mTrackEtaPhiNCls[nHisto].get());
  }
  objar.Add(mCATrackEta.get());
  objar.Add(mLTFTrackEta.get());
  objar.Add(mTrackTanl.get());

  objar.Add(mTrackROFNEntries.get());
  objar.Add(mClusterROFNEntries.get());

  objar.Add(mClusterSensorIndex.get());
  objar.Add(mClusterPatternIndex.get());

  if (mUseMC) {
    objar.Add(mHistPhiRecVsPhiGen.get());
    objar.Add(mHistEtaRecVsEtaGen.get());
    for (int TrackType = 0; TrackType < kNumberOfTrackTypes; TrackType++) {
      objar.Add(mHistPhiVsEta[TrackType].get());
      objar.Add(mHistPtVsEta[TrackType].get());
      objar.Add(mHistPhiVsPt[TrackType].get());
      objar.Add(mHistZvtxVsEta[TrackType].get());
      if (TrackType == kGen || TrackType == kTrackable) {
        objar.Add(mHistRVsZ[TrackType].get());
      }
    }

    // Histos for Reconstruction assessment

    for (auto& h : mTH3Histos) {
      objar.Add(h.get());
    }

    if (mFinalizeAnalysis) {
      objar.Add(mHistVxtOffsetProjection.get());
    }

    objar.Add(mChargeMatchEff.get());

    if (mFinalizeAnalysis) {
      for (int slicedCanvas = 0; slicedCanvas < kNSlicedTH3; slicedCanvas++) {
        objar.Add(mSlicedCanvas[slicedCanvas]);
      }
    }
  }
}

//__________________________________________________________
void MFTAssessment::TH3Slicer(TCanvas* canvas, std::unique_ptr<TH3F>& histo3D, std::vector<float> list, double window, int iPar, float marker_size)
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
    for (auto etamax : list) {
      auto etamin = etamax + window;
      histo3D->GetYaxis()->SetRangeUser(etamin, etamax);
      std::string ytitle = "\\sigma (" + std::string(histo3D->GetZaxis()->GetTitle()) + ")";
      auto title = Form("_%1.2f_%1.2f_xz", etamin, etamax);
      auto aDBG = (TH2F*)histo3D->Project3D(title);
      aDBG->FitSlicesX(nullptr, 0, -1, 4, "QNR", &aSlices);
      auto th1DBG = (TH1F*)aSlices[iPar];
      th1DBG->SetTitle(Form("%1.2f > \\eta > %1.2f", etamax, etamin));
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

//__________________________________________________________
bool MFTAssessment::loadHistos()
{
  if (mFinalizeAnalysis) {
    throw std::runtime_error("MFTAssessment error: data already loaded");
  }
  mFinalizeAnalysis = true;

  TObjArray* objar;

  TFile* f = new TFile(Form("MFTAssessment.root"));

  mTrackNumberOfClusters = std::unique_ptr<TH1F>((TH1F*)f->Get("mMFTTrackNumberOfClusters"));

  mCATrackNumberOfClusters = std::unique_ptr<TH1F>((TH1F*)f->Get("mMFTCATrackNumberOfClusters"));

  mLTFTrackNumberOfClusters = std::unique_ptr<TH1F>((TH1F*)f->Get("mMFTLTFTrackNumberOfClusters"));

  mTrackOnvQPt = std::unique_ptr<TH1F>((TH1F*)f->Get("mMFTTrackOnvQPt"));

  mTrackChi2 = std::unique_ptr<TH1F>((TH1F*)f->Get("mMFTTrackChi2"));

  mTrackCharge = std::unique_ptr<TH1F>((TH1F*)f->Get("mMFTTrackCharge"));

  mTrackPhi = std::unique_ptr<TH1F>((TH1F*)f->Get("mMFTTrackPhi"));

  mPositiveTrackPhi = std::unique_ptr<TH1F>((TH1F*)f->Get("mMFTPositiveTrackPhi"));

  mNegativeTrackPhi = std::unique_ptr<TH1F>((TH1F*)f->Get("mMFTNegativeTrackPhi"));

  mTrackEta = std::unique_ptr<TH1F>((TH1F*)f->Get("mMFTTrackEta"));

  //---------------------------------------------------------------------------

  mMFTClsZ = std::unique_ptr<TH1F>((TH1F*)f->Get("mMFTClsZ"));

  mMFTClsOfTracksZ = std::unique_ptr<TH1F>((TH1F*)f->Get("mMFTClsOfTracksZ"));

  for (auto nMFTLayer = 0; nMFTLayer < 10; nMFTLayer++) {
    mMFTClsXYinLayer[nMFTLayer] = std::unique_ptr<TH2F>((TH2F*)f->Get(Form("mMFTClsXYinLayer%d", nMFTLayer)));
    mMFTClsOfTracksXYinLayer[nMFTLayer] = std::unique_ptr<TH2F>((TH2F*)f->Get(Form("mMFTClsOfTracksXYinLayer%d", nMFTLayer)));
  }

  for (auto nMFTDisk = 0; nMFTDisk < 5; nMFTDisk++) {
    mMFTClsXYRedundantInDisk[nMFTDisk] = std::unique_ptr<TH2F>((TH2F*)f->Get(Form("mMFTClsXYRedundantInDisk%d", nMFTDisk)));
  }

  for (auto minNClusters : sMinNClustersList) {
    auto nHisto = minNClusters - sMinNClustersList[0];
    mTrackEtaNCls[nHisto] = std::unique_ptr<TH1F>((TH1F*)f->Get(Form("mMFTTrackEta_%d_MinClusters", minNClusters)));

    mTrackPhiNCls[nHisto] = std::unique_ptr<TH1F>((TH1F*)f->Get(Form("mMFTTrackPhi_%d_MinClusters", minNClusters)));

    mTrackXYNCls[nHisto] = std::unique_ptr<TH2F>((TH2F*)f->Get(Form("mMFTTrackXY_%d_MinClusters", minNClusters)));

    mTrackEtaPhiNCls[nHisto] = std::unique_ptr<TH2F>((TH2F*)f->Get(Form("mMFTTrackEtaPhi_%d_MinClusters", minNClusters)));
  }

  mCATrackEta = std::unique_ptr<TH1F>((TH1F*)f->Get("mMFTCATrackEta"));

  mLTFTrackEta = std::unique_ptr<TH1F>((TH1F*)f->Get("mMFTLTFTrackEta"));

  mTrackTanl = std::unique_ptr<TH1F>((TH1F*)f->Get("mMFTTrackTanl"));

  mClusterROFNEntries = std::unique_ptr<TH1F>((TH1F*)f->Get("mMFTClustersROFSize"));

  mTrackROFNEntries = std::unique_ptr<TH1F>((TH1F*)f->Get("mMFTTrackROFSize"));

  mTracksBC = std::unique_ptr<TH1F>((TH1F*)f->Get("mMFTTracksBC"));

  mNOfTracksTime = std::unique_ptr<TH1F>((TH1F*)f->Get("mNOfTracksTime"));

  mNOfClustersTime = std::unique_ptr<TH1F>((TH1F*)f->Get("mNOfClustersTime"));

  mClusterSensorIndex = std::unique_ptr<TH1F>((TH1F*)f->Get("mMFTClusterSensorIndex"));

  mClusterPatternIndex = std::unique_ptr<TH1F>((TH1F*)f->Get("mMFTClusterPatternIndex"));

  // Creating MC-based histos
  if (mUseMC) {

    mHistPhiRecVsPhiGen = std::unique_ptr<TH2F>((TH2F*)f->Get("mHistPhiRecVsPhiGen"));

    mHistEtaRecVsEtaGen = std::unique_ptr<TH2F>((TH2F*)f->Get("mHistEtaRecVsEtaGen"));

    for (int trackType = 0; trackType < kNumberOfTrackTypes; trackType++) {
      mHistPhiVsEta[trackType] = std::unique_ptr<TH2F>((TH2F*)f->Get((std::string("mHistPhiVsEta") + mNameOfTrackTypes[trackType]).c_str()));

      mHistPtVsEta[trackType] = std::unique_ptr<TH2F>((TH2F*)f->Get((std::string("mHistPtVsEta") + mNameOfTrackTypes[trackType]).c_str()));

      mHistPhiVsPt[trackType] = std::unique_ptr<TH2F>((TH2F*)f->Get((std::string("mHistPhiVsPt") + mNameOfTrackTypes[trackType]).c_str()));

      if (trackType != kReco) {
        mHistZvtxVsEta[trackType] = std::unique_ptr<TH2F>((TH2F*)f->Get((std::string("mHistZvtxVsEta") + mNameOfTrackTypes[trackType]).c_str()));
      }
      if (trackType == kGen || trackType == kTrackable) {
        mHistRVsZ[trackType] = std::unique_ptr<TH2F>((TH2F*)f->Get((std::string("mHistRVsZ") + mNameOfTrackTypes[trackType]).c_str()));
      }
    }

    // Histos for Reconstruction assessment
    mChargeMatchEff = std::unique_ptr<TEfficiency>((TEfficiency*)f->Get("QMatchEff"));

    const int nTH3Histos = TH3Names.size();
    auto n3Histo = 0;
    for (auto& h : mTH3Histos) {
      h = std::unique_ptr<TH3F>((TH3F*)f->Get(TH3Names[n3Histo]));
      ++n3Histo;
    }
  }
  return true;
}

//__________________________________________________________
void MFTAssessment::finalizeAnalysis()
{
  mHistVxtOffsetProjection = std::unique_ptr<TH2F>((TH2F*)mTH3Histos[kTH3TrackDeltaXDeltaYEta]->Project3D("colz yz"));
  mHistVxtOffsetProjection->SetNameTitle("Vertex_XY_OffSet", "Track-MC_Vertex Offset");
  mHistVxtOffsetProjection->SetOption("COLZ");

  if (mFinalizeAnalysis) {
    std::vector<float> ptList({.5, 1.5, 5., 10., 15., 18.0});
    float ptWindow = 0.4;
    std::vector<float> etaList({-2.5, -2.8, -3.1});
    float etaWindow = -0.2;

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
      TH3Slicer(mSlicedCanvas[nCanvas], mTH3Histos[TH3SlicedMap[nCanvas]], sliceList, sliceWindow, 2);
    }
  }
}
