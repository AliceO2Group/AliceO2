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
#include "DetectorsBase/GeometryManager.h"
#include <Framework/InputRecord.h>

using namespace o2::mft;

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
  }
}

//__________________________________________________________
bool MFTAssessment::init()
{
  auto MaxClusterROFSize = 5000;
  auto MaxTrackROFSize = 1000;
  auto ROFLengthInBC = 198;
  auto ROFsPerOrbit = o2::constants::lhc::LHCMaxBunches / ROFLengthInBC;
  auto MaxDuration = 60.f;
  auto TimeBinSize = 0.01f;
  auto NofTimeBins = static_cast<int>(MaxDuration / TimeBinSize);

  // Creating data-only histos
  mTrackNumberOfClusters = std::make_unique<TH1F>("tracks/mMFTTrackNumberOfClusters",
                                                  "Number Of Clusters Per Track; # clusters; # entries", 10, 0.5, 10.5);

  mCATrackNumberOfClusters = std::make_unique<TH1F>("tracks/CA/mMFTCATrackNumberOfClusters",
                                                    "Number Of Clusters Per CA Track; # clusters; # tracks", 10, 0.5, 10.5);

  mLTFTrackNumberOfClusters = std::make_unique<TH1F>("tracks/LTF/mMFTLTFTrackNumberOfClusters",
                                                     "Number Of Clusters Per LTF Track; # clusters; # entries", 10, 0.5, 10.5);

  mTrackOnvQPt = std::make_unique<TH1F>("tracks/mMFTTrackOnvQPt", "Track q/p_{T}; q/p_{T} [1/GeV]; # entries", 50, -2, 2);

  mTrackChi2 = std::make_unique<TH1F>("tracks/mMFTTrackChi2", "Track #chi^{2}; #chi^{2}; # entries", 21, -0.5, 20.5);

  mTrackCharge = std::make_unique<TH1F>("tracks/mMFTTrackCharge", "Track Charge; q; # entries", 3, -1.5, 1.5);

  mTrackPhi = std::make_unique<TH1F>("tracks/mMFTTrackPhi", "Track #phi; #phi; # entries", 100, -3.2, 3.2);

  mPositiveTrackPhi = std::make_unique<TH1F>("tracks/mMFTPositiveTrackPhi", "Positive Track #phi; #phi; # entries", 100, -3.2, 3.2);

  mNegativeTrackPhi = std::make_unique<TH1F>("tracks/mMFTNegativeTrackPhi", "Negative Track #phi; #phi; # entries", 100, -3.2, 3.2);

  mTrackEta = std::make_unique<TH1F>("tracks/mMFTTrackEta", "Track #eta; #eta; # entries", 50, -4, -2);

  for (auto minNClusters : sMinNClustersList) {
    auto nHisto = minNClusters - sMinNClustersList[0];
    mTrackEtaNCls[nHisto] = std::make_unique<TH1F>(Form("tracks/mMFTTrackEta_%d_MinClusters", minNClusters), Form("Track #eta (NCls >= %d); #eta; # entries", minNClusters), 50, -4, -2);

    mTrackPhiNCls[nHisto] = std::make_unique<TH1F>(Form("tracks/mMFTTrackPhi_%d_MinClusters", minNClusters), Form("Track #phi (NCls >= %d); #phi; # entries", minNClusters), 100, -3.2, 3.2);

    mTrackXYNCls[nHisto] = std::make_unique<TH2F>(Form("tracks/mMFTTrackXY_%d_MinClusters", minNClusters), Form("Track Position (NCls >= %d); x; y", minNClusters), 320, -16, 16, 320, -16, 16);
    mTrackXYNCls[nHisto]->SetOption("COLZ");

    mTrackEtaPhiNCls[nHisto] = std::make_unique<TH2F>(Form("tracks/mMFTTrackEtaPhi_%d_MinClusters", minNClusters), Form("Track #eta , #phi (NCls >= %d); #eta; #phi", minNClusters), 50, -4, -2, 100, -3.2, 3.2);
    mTrackEtaPhiNCls[nHisto]->SetOption("COLZ");
  }

  mCATrackEta = std::make_unique<TH1F>("tracks/CA/mMFTCATrackEta", "CA Track #eta; #eta; # entries", 50, -4, -2);

  mLTFTrackEta = std::make_unique<TH1F>("tracks/LTF/mMFTLTFTrackEta", "LTF Track #eta; #eta; # entries", 50, -4, -2);

  mTrackTanl = std::make_unique<TH1F>("tracks/mMFTTrackTanl", "Track tan #lambda; tan #lambda; # entries", 100, -25, 0);

  mClusterROFNEntries = std::make_unique<TH1F>("clusters/mMFTClustersROFSize", "MFT Cluster ROFs size; ROF Size; # entries", MaxClusterROFSize, 0, MaxClusterROFSize);

  mTrackROFNEntries = std::make_unique<TH1F>("tracks/mMFTTrackROFSize", "MFT Track ROFs size; ROF Size; # entries", MaxTrackROFSize, 0, MaxTrackROFSize);

  mTracksBC = std::make_unique<TH1F>("tracks/mMFTTracksBC", "Tracks per BC (sum over orbits); BCid; # entries", ROFsPerOrbit, 0, o2::constants::lhc::LHCMaxBunches);
  mTracksBC->SetMinimum(0.1);

  mNOfTracksTime = std::make_unique<TH1F>("tracks/mNOfTracksTime", "Number of tracks per time bin; time (s); # entries", NofTimeBins, 0, MaxDuration);
  mNOfTracksTime->SetMinimum(0.1);

  mNOfClustersTime = std::make_unique<TH1F>("clusters/mNOfClustersTime", "Number of clusters per time bin; time (s); # entries", NofTimeBins, 0, MaxDuration);
  mNOfClustersTime->SetMinimum(0.1);

  mClusterSensorIndex = std::make_unique<TH1F>("clusters/mMFTClusterSensorIndex", "Chip Cluster Occupancy;Chip ID;#Entries", 936, -0.5, 935.5);

  mClusterPatternIndex = std::make_unique<TH1F>("clusters/mMFTClusterPatternIndex", "Cluster Pattern ID;Pattern ID;#Entries", 300, -0.5, 299.5);

  // Creating MC-based histos
  if (mUseMC) {

    LOG(info) << "Initializing MC Reader";
    if (!mcReader.initFromDigitContext("collisioncontext.root")) {
      throw std::invalid_argument("initialization of MCKinematicsReader failed");
    }

    mHistPhiRecVsPhiGen = std::make_unique<TH2F>("mc/mHistPhiRecVsPhiGen", "Phi Rec Vs Phi Gen of true reco tracks ", 24, -0.5 * TMath::Pi(), 0.5 * TMath::Pi(), 24, -0.5 * TMath::Pi(), 0.5 * TMath::Pi());
    mHistPhiRecVsPhiGen->SetXTitle((std::string("#phi of ") + mNameOfTrackTypes[kGen]).c_str());
    mHistPhiRecVsPhiGen->SetYTitle((std::string("#phi of ") + mNameOfTrackTypes[kRecoTrue]).c_str());
    mHistPhiRecVsPhiGen->Sumw2();
    mHistPhiRecVsPhiGen->SetOption("COLZ");

    mHistEtaRecVsEtaGen = std::make_unique<TH2F>("mc/mHistEtaRecVsEtaGen", "Eta Rec Vs Eta Gen of true reco tracks ", 35, 1.0, 4.5, 35, 1.0, 4.5);
    mHistEtaRecVsEtaGen->SetXTitle((std::string("#eta of ") + mNameOfTrackTypes[kGen]).c_str());
    mHistEtaRecVsEtaGen->SetYTitle((std::string("#eta of ") + mNameOfTrackTypes[kRecoTrue]).c_str());
    mHistEtaRecVsEtaGen->Sumw2();
    mHistEtaRecVsEtaGen->SetOption("COLZ");

    for (int trackType = 0; trackType < kNumberOfTrackTypes; trackType++) {
      // mHistPhiVsEta
      mHistPhiVsEta[trackType] = std::make_unique<TH2F>((std::string("mc/mHistPhiVsEta") + mNameOfTrackTypes[trackType]).c_str(), (std::string("Phi Vs Eta of ") + mNameOfTrackTypes[trackType]).c_str(), 35, 1.0, 4.5, 24, -0.5 * TMath::Pi(), 0.5 * TMath::Pi());
      mHistPhiVsEta[trackType]->SetXTitle((std::string("#eta of ") + mNameOfTrackTypes[trackType]).c_str());
      mHistPhiVsEta[trackType]->SetYTitle((std::string("#phi of ") + mNameOfTrackTypes[trackType]).c_str());
      mHistPhiVsEta[trackType]->Sumw2();
      mHistPhiVsEta[trackType]->SetOption("COLZ");

      // mHistPtVsEta
      mHistPtVsEta[trackType] = std::make_unique<TH2F>((std::string("mc/mHistPtVsEta") + mNameOfTrackTypes[trackType]).c_str(), (std::string("Pt Vs Eta of ") + mNameOfTrackTypes[trackType]).c_str(), 35, 1.0, 4.5, 40, 0., 10.);
      mHistPtVsEta[trackType]->SetXTitle((std::string("#eta of ") + mNameOfTrackTypes[trackType]).c_str());
      mHistPtVsEta[trackType]->SetYTitle((std::string("p_{T} (GeV/c) of ") + mNameOfTrackTypes[trackType]).c_str());
      mHistPtVsEta[trackType]->Sumw2();
      mHistPtVsEta[trackType]->SetOption("COLZ");

      // mHistPhiVsPt
      mHistPhiVsPt[trackType] = std::make_unique<TH2F>((std::string("mc/mHistPhiVsPt") + mNameOfTrackTypes[trackType]).c_str(), (std::string("Phi Vs Pt of ") + mNameOfTrackTypes[trackType]).c_str(), 40, 0., 10., 24, -0.5 * TMath::Pi(), 0.5 * TMath::Pi());
      mHistPhiVsPt[trackType]->SetXTitle((std::string("p_{T} (GeV/c) of ") + mNameOfTrackTypes[trackType]).c_str());
      mHistPhiVsPt[trackType]->SetYTitle((std::string("#phi of ") + mNameOfTrackTypes[trackType]).c_str());
      mHistPhiVsPt[trackType]->Sumw2();
      mHistPhiVsPt[trackType]->SetOption("COLZ");

      if (trackType != kReco) {
        // mHistZvtxVsEta
        mHistZvtxVsEta[trackType] = std::make_unique<TH2F>((std::string("mc/mHistZvtxVsEta") + mNameOfTrackTypes[trackType]).c_str(), (std::string("Z_{vtx} Vs Eta of ") + mNameOfTrackTypes[trackType]).c_str(), 35, 1.0, 4.5, 15, -15, 15);
        mHistZvtxVsEta[trackType]->SetXTitle((std::string("#eta of ") + mNameOfTrackTypes[trackType]).c_str());
        mHistZvtxVsEta[trackType]->SetYTitle((std::string("z_{vtx} (cm) of ") + mNameOfTrackTypes[trackType]).c_str());
        mHistZvtxVsEta[trackType]->Sumw2();
        mHistZvtxVsEta[trackType]->SetOption("COLZ");
      }
      // mHistRVsZ]
      if (trackType == kGen || trackType == kTrackable) {
        mHistRVsZ[trackType] = std::make_unique<TH2F>((std::string("mc/mHistRVsZ") + mNameOfTrackTypes[trackType]).c_str(), (std::string("R Vs Z of ") + mNameOfTrackTypes[trackType]).c_str(), 400, -80., 20., 400, 0., 80.);
        mHistRVsZ[trackType]->SetXTitle((std::string("z (cm) origin of ") + mNameOfTrackTypes[trackType]).c_str());
        mHistRVsZ[trackType]->SetYTitle((std::string("R (cm) radius of origin of ") + mNameOfTrackTypes[trackType]).c_str());
        mHistRVsZ[trackType]->Sumw2();
        mHistRVsZ[trackType]->SetOption("COLZ");
      }
    }
  }

  return true;
}

//__________________________________________________________
void MFTAssessment::runASyncQC(o2::framework::ProcessingContext& ctx)
{

  // get tracks
  mMFTTracks = ctx.inputs().get<gsl::span<o2::mft::TrackMFT>>("tracks");
  mMFTTracksROF = ctx.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("tracksrofs");

  // get clusters
  mMFTClusters = ctx.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("compClusters");
  mMFTClustersROF = ctx.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("clustersrofs");

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

  for (auto& oneCluster : mMFTClusters) {
    mClusterSensorIndex->Fill(oneCluster.getSensorID());
    mClusterPatternIndex->Fill(oneCluster.getPatternID());
  }

  // fill the tracks histogram

  for (const auto& rof : mMFTTracksROF) {
    mTrackROFNEntries->Fill(rof.getNEntries());
    mTracksBC->Fill(rof.getBCData().bc, rof.getNEntries());
    float seconds = orbitToSeconds(rof.getBCData().orbit, mRefOrbit) + rof.getBCData().bc * o2::constants::lhc::LHCBunchSpacingNS * 1e-9;
    mNOfTracksTime->Fill(seconds, rof.getNEntries());
  }

  for (auto& oneTrack : mMFTTracks) {
    mTrackNumberOfClusters->Fill(oneTrack.getNumberOfPoints());
    mTrackChi2->Fill(oneTrack.getTrackChi2());
    mTrackCharge->Fill(oneTrack.getCharge());
    mTrackPhi->Fill(oneTrack.getPhi());
    mTrackEta->Fill(oneTrack.getEta());
    mTrackTanl->Fill(oneTrack.getTanl());

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
  for (auto src = 0; src < mcReader.getNSources(); src++) {
    for (Int_t event = 0; event < mcReader.getNEvents(src); event++) {
      const auto& mcTracks = mcReader.getTracks(src, event);
      for (const auto& mcParticle : mcTracks) {
        auto evh = mcReader.getMCEventHeader(src, event);
        addMCParticletoHistos(&mcParticle, kGen, evh);
      } // mcTracks
    }   // events
  }     // sources
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

  for (auto& trackClsInDisk : mcTrackHasClusterInMFTDisks) {
    auto& clsdisk = trackClsInDisk.second;
    auto nMFTDisks = 0;
    for (auto disk : {0, 1, 2, 3, 4}) {
      nMFTDisks += int(clsdisk[disk]);
    }
    if (nMFTDisks >= 4) {
      mMFTTrackables[trackClsInDisk.first] = true;
      auto const* mcParticle = mcReader.getTrack(trackClsInDisk.first);
      srcID = trackClsInDisk.first.getSourceID();
      evnID = trackClsInDisk.first.getEventID();
      auto evH = mcReader.getMCEventHeader(srcID, evnID);
      addMCParticletoHistos(mcParticle, kTrackable, evH);
    }
  }
}

//__________________________________________________________
void MFTAssessment::addMCParticletoHistos(const MCTrack* mcTr, const int TrackType, const o2::dataformats::MCEventHeader& evH)
{
  auto zVtx = evH.GetZ();

  auto pt = mcTr->GetPt();
  auto eta = -1 * mcTr->GetEta();
  auto phi = mcTr->GetPhi();
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
void MFTAssessment::processRecoAndTrueTracks()
{
  auto trkId = 0;
  for (auto& mftTrack : mMFTTracks) {
    auto pt = mftTrack.getPt();
    auto eta = -1 * mftTrack.getEta();
    auto phi = mftTrack.getPhi();
    auto TrackType = kReco;
    mHistPtVsEta[TrackType]->Fill(eta, pt);
    mHistPhiVsEta[TrackType]->Fill(eta, phi);
    mHistPhiVsPt[TrackType]->Fill(pt, phi);

    auto trackLabel = mMFTTrackLabels[trkId];
    if (trackLabel.isCorrect()) {
      TrackType = kRecoTrue;
      auto evH = mcReader.getMCEventHeader(trackLabel.getSourceID(), trackLabel.getEventID());
      auto zVtx = evH.GetZ();
      mHistPtVsEta[TrackType]->Fill(eta, pt);
      mHistPhiVsEta[TrackType]->Fill(eta, phi);
      mHistPhiVsPt[TrackType]->Fill(pt, phi);
      mHistZvtxVsEta[TrackType]->Fill(eta, zVtx);

      auto const* mcParticle = mcReader.getTrack(trackLabel);
      auto etaGen = -1 * mcParticle->GetEta();
      auto phiGen = mcParticle->GetPhi();
      mHistPhiRecVsPhiGen->Fill(phiGen, phi); // This should be at same Z
      mHistEtaRecVsEtaGen->Fill(etaGen, eta);
    }
    trkId++;
  }
}

//__________________________________________________________
void MFTAssessment::finalize()
{
}

//__________________________________________________________
void MFTAssessment::getHistos(TObjArray& objar)
{
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
  }
}
