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

/// \file MatchGlobalFwdAssessment.h
/// \brief Class to perform assessment of GlobalForward Tracking
/// \author rafael.pezzi at cern.ch

#ifndef ALICEO2_GLOFWD_ASSESSMENT
#define ALICEO2_GLOFWD_ASSESSMENT

#include <TH1F.h>
#include <TH2F.h>
#include <TH3F.h>
#include <TCanvas.h>
#include <TEfficiency.h>
#include <TObjArray.h>
#include "Framework/ProcessingContext.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTrack.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "DataFormatsMCH/TrackMCH.h"
#include <DataFormatsITSMFT/ROFRecord.h>
#include <DataFormatsITSMFT/CompCluster.h>
#include "ReconstructionDataFormats/GlobalFwdTrack.h"
#include "Steer/MCKinematicsReader.h"
#include <unordered_map>
#include <vector>

namespace o2
{

namespace globaltracking
{

enum mMFTTrackTypes { kReco,
                      kGen,
                      kPairable,
                      kRecoTrue,
                      kNumberOfTrackTypes };

using ClusterLabelsType = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
using TrackLabelsType = std::vector<o2::MCCompLabel>;
using MCTrack = o2::MCTrackT<float>;

class GloFwdAssessment
{
 public:
  GloFwdAssessment() = delete;
  GloFwdAssessment(bool useMC) : mUseMC(useMC){};
  ~GloFwdAssessment() = default;
  void disableMIDFilter() { mMIDFilterEnabled = false; }

  void init(bool finalizeAnalysis);
  void createHistos();
  void loadHistos();
  void deleteHistograms();

  void reset();

  void runBasicQC(o2::framework::ProcessingContext& ctx);
  void processPairables();
  void processGeneratedTracks();
  void processRecoTracks();
  void processTrueTracks();
  void fillTrueRecoTracksMap()
  {
    mTrueTracksMap.resize(mcReader.getNSources());
    auto src = 0;
    for (auto& map : mTrueTracksMap) {
      map.resize(mcReader.getNEvents(src++));
    }
    auto id = 0;
    for (const auto& trackLabel : mFwdTrackLabels) {
      if (trackLabel.isCorrect()) {
        mTrueTracksMap[trackLabel.getSourceID()][trackLabel.getEventID()].push_back(id);
      }
      id++;
    }
  }
  void addMCParticletoHistos(const MCTrack* mcTr, const int TrackType, const o2::dataformats::MCEventHeader& evH);

  void finalizeAnalysis();
  void finalizeRecoAndPairables();
  void finalizePurityAndEff();
  void finalizeCutConfig(float minCut, float maxCut, int nSteps)
  {
    mFinalizeMinCut = minCut;
    mFinalizeMaxCut = maxCut;
    mNFinalizeSteps = nSteps;
  }

  void getHistos(TObjArray& objar);
  void setBz(float bz) { mBz = bz; }

  double orbitToSeconds(uint32_t orbit, uint32_t refOrbit)
  {
    return (orbit - refOrbit) * o2::constants::lhc::LHCOrbitNS / 1E9;
  }

 private:
  gsl::span<const o2::dataformats::GlobalFwdTrack> mGlobalFwdTracks;
  gsl::span<const o2::mft::TrackMFT> mMFTTracks;
  gsl::span<const o2::mch::TrackMCH> mMCHTracks;
  gsl::span<const o2::itsmft::ROFRecord> mMFTTracksROF;
  gsl::span<const o2::itsmft::CompClusterExt> mMFTClusters;
  gsl::span<const o2::itsmft::ROFRecord> mMFTClustersROF;

  // MC Labels
  bool mUseMC = false;

  gsl::span<const o2::MCCompLabel> mMFTTrackLabels;
  gsl::span<const o2::MCCompLabel> mMCHTrackLabels;
  gsl::span<const o2::MCCompLabel> mFwdTrackLabels;

  o2::steer::MCKinematicsReader mcReader; // reader of MC information

  // Histos for reconstructed tracks
  std::unique_ptr<TH1F> mTrackNumberOfClusters = nullptr;
  std::unique_ptr<TH1F> mTrackInvQPt = nullptr;
  std::unique_ptr<TH1F> mTrackChi2 = nullptr;
  std::unique_ptr<TH1F> mTrackCharge = nullptr;
  std::unique_ptr<TH1F> mTrackPhi = nullptr;
  std::unique_ptr<TH1F> mTrackEta = nullptr;
  std::array<std::unique_ptr<TH1F>, 7> mTrackEtaNCls = {nullptr};
  std::array<std::unique_ptr<TH1F>, 7> mTrackPhiNCls = {nullptr};
  std::array<std::unique_ptr<TH2F>, 7> mTrackXYNCls = {nullptr};
  std::array<std::unique_ptr<TH2F>, 7> mTrackEtaPhiNCls = {nullptr};
  std::unique_ptr<TH1F> mTrackCotl = nullptr;

  // Histos and data for MC analysis
  std::vector<std::string> mNameOfTrackTypes = {"Rec",
                                                "Gen",
                                                "Pairable",
                                                "RecoTrue"};

  std::unique_ptr<TH2F> mHistPhiRecVsPhiGen = nullptr;
  std::unique_ptr<TH2F> mHistEtaRecVsEtaGen = nullptr;

  std::array<std::unique_ptr<TH2F>, kNumberOfTrackTypes> mHistPhiVsEta;
  std::array<std::unique_ptr<TH2F>, kNumberOfTrackTypes> mHistPtVsEta;
  std::array<std::unique_ptr<TH2F>, kNumberOfTrackTypes> mHistPhiVsPt;
  std::array<std::unique_ptr<TH2F>, kNumberOfTrackTypes> mHistZvtxVsEta;
  std::array<std::unique_ptr<TH2F>, kNumberOfTrackTypes> mHistRVsZ;

  // Histos for reconstruction assessment

  std::unique_ptr<TEfficiency> mChargeMatchEff = nullptr;
  std::unique_ptr<TH2D> mPairingEtaPt = nullptr;
  std::unique_ptr<TH2D> mTruePairingEtaPt = nullptr;
  std::unique_ptr<TH2F> mHistVxtOffsetProjection = nullptr;

  std::vector<std::unique_ptr<TH2D>> mPurityPtInnerVecTH2;
  std::vector<std::unique_ptr<TH2D>> mPurityPtOuterVecTH2;
  std::vector<std::unique_ptr<TH1D>> mPairingPtInnerVecTH1;
  std::vector<std::unique_ptr<TH1D>> mPairingPtOuterVecTH1;
  std::vector<std::unique_ptr<TH1D>> mTruePairingPtInnerVecTH1;
  std::vector<std::unique_ptr<TH1D>> mTruePairingPtOuterVecTH1;
  std::vector<std::unique_ptr<TH2D>> mPairingEtaPtVec;
  std::vector<std::unique_ptr<TH2D>> mTruePairingEtaPtVec;

  enum TH3HistosCodes {
    kTH3GMTrackDeltaXDeltaYEta,
    kTH3GMTrackDeltaXDeltaYPt,
    kTH3GMTrackDeltaXVertexPtEta,
    kTH3GMTrackDeltaYVertexPtEta,
    kTH3GMTrackInvQPtResolutionPtEta,
    kTH3GMTrackInvQPtResMCHPtEta,
    kTH3GMTrackXPullPtEta,
    kTH3GMTrackYPullPtEta,
    kTH3GMTrackPhiPullPtEta,
    kTH3GMTrackCotlPullPtEta,
    kTH3GMTrackInvQPtPullPtEta,
    kTH3GMTrackReducedChi2PtEta,
    kTH3GMTrackPtEtaChi2,
    kTH3GMTrackPtEtaMatchScore,
    kTH3GMTruePtEtaChi2,
    kTH3GMTruePtEtaMatchScore,
    kTH3GMTruePtEtaMatchScore_MC,
    kTH3GMCloseMatchPtEtaChi2,
    kTH3GMCloseMatchPtEtaMatchScore,
    kTH3GMPairablePtEtaZ,
    kNTH3Histos
  };

  std::map<int, const char*> TH3Names{
    {kTH3GMTrackDeltaXDeltaYEta, "TH3GMTrackDeltaXDeltaYEta"},
    {kTH3GMTrackDeltaXDeltaYPt, "TH3GMTrackDeltaXDeltaYPt"},
    {kTH3GMTrackDeltaXVertexPtEta, "TH3GMTrackDeltaXVertexPtEta"},
    {kTH3GMTrackDeltaYVertexPtEta, "TH3GMTrackDeltaYVertexPtEta"},
    {kTH3GMTrackInvQPtResolutionPtEta, "TH3GMTrackInvQPtResolutionPtEta"},
    {kTH3GMTrackInvQPtResMCHPtEta, "TH3GMTrackInvQPtResMCHPtEta"},
    {kTH3GMTrackXPullPtEta, "TH3GMTrackXPullPtEta"},
    {kTH3GMTrackYPullPtEta, "TH3GMTrackYPullPtEta"},
    {kTH3GMTrackPhiPullPtEta, "TH3GMTrackPhiPullPtEta"},
    {kTH3GMTrackCotlPullPtEta, "TH3GMTrackCotlPullPtEta"},
    {kTH3GMTrackInvQPtPullPtEta, "TH3GMTrackInvQPtPullPtEta"},
    {kTH3GMTrackReducedChi2PtEta, "TH3GMTrackReducedChi2PtEta"},
    {kTH3GMCloseMatchPtEtaChi2, "TH3GMCloseMatchPtEtaChi2"},
    {kTH3GMCloseMatchPtEtaMatchScore, "TH3GMCloseMatchPtEtaMatchScore"},
    {kTH3GMPairablePtEtaZ, "TH3GMPairablePtEtaZ"},
    {kTH3GMTrackPtEtaChi2, "TH3GMTrackPtEtaChi2"},
    {kTH3GMTrackPtEtaMatchScore, "TH3GMTrackPtEtaMatchScore"},
    {kTH3GMTruePtEtaChi2, "TH3GMTruePtEtaChi2"},
    {kTH3GMTruePtEtaMatchScore, "TH3GMTruePtEtaMatchScore"},
    {kTH3GMTruePtEtaMatchScore_MC, "TH3GMTruePtEtaMatchScore_MC"}};

  std::map<int, const char*> TH3Titles{
    {kTH3GMTrackDeltaXDeltaYEta, "TH3GMTrackDeltaXDeltaYEta"},
    {kTH3GMTrackDeltaXDeltaYPt, "TH3GMTrackDeltaXDeltaYPt"},
    {kTH3GMTrackDeltaXVertexPtEta, "TH3GMTrackDeltaXVertexPtEta"},
    {kTH3GMTrackDeltaYVertexPtEta, "TH3GMTrackDeltaYVertexPtEta"},
    {kTH3GMTrackInvQPtResolutionPtEta, "TH3GMTrackInvQPtResolutionPtEta"},
    {kTH3GMTrackInvQPtResMCHPtEta, "TH3GMTrackInvQPtResMCHPtEta"},
    {kTH3GMTrackXPullPtEta, "TH3GMTrackXPullPtEta"},
    {kTH3GMTrackYPullPtEta, "TH3GMTrackYPullPtEta"},
    {kTH3GMTrackPhiPullPtEta, "TH3GMTrackPhiPullPtEta"},
    {kTH3GMTrackCotlPullPtEta, "TH3GMTrackCotlPullPtEta"},
    {kTH3GMTrackInvQPtPullPtEta, "TH3GMTrackInvQPtPullPtEta"},
    {kTH3GMTrackReducedChi2PtEta, "TH3GMTrackReducedChi2PtEta"},
    {kTH3GMCloseMatchPtEtaChi2, "TH3GMCloseMatchPtEtaChi2"},
    {kTH3GMCloseMatchPtEtaMatchScore, "TH3GMCloseMatchPtEtaMatchScore"},
    {kTH3GMPairablePtEtaZ, "TH3GMPairablePtEtaZ"},
    {kTH3GMTrackPtEtaChi2, "TH3GMTrackPtEtaChi2"},
    {kTH3GMTrackPtEtaMatchScore, "TH3GMTrackPtEtaMatchScore"},
    {kTH3GMTruePtEtaChi2, "TH3GMTruePtEtaChi2"},
    {kTH3GMTruePtEtaMatchScore, "TH3GMTruePtEtaMatchScore"},
    {kTH3GMTruePtEtaMatchScore_MC, "TH3GMTruePtEtaMatchScore_MC"}};

  std::map<int, std::array<double, 9>> TH3Binning{
    {kTH3GMTrackDeltaXDeltaYEta, {16, 2.2, 3.8, 1000, -1000, 1000, 1000, -1000, 1000}},
    {kTH3GMTrackDeltaXDeltaYPt, {40, 0, 20, 1000, -1000, 1000, 1000, -1000, 1000}},
    {kTH3GMTrackDeltaYVertexPtEta, {40, 0, 20, 16, 2.2, 3.8, 1000, -1000, 1000}},
    {kTH3GMTrackDeltaXVertexPtEta, {40, 0, 20, 16, 2.2, 3.8, 1000, -1000, 1000}},
    {kTH3GMTrackInvQPtResolutionPtEta, {40, 0, 20, 16, 2.2, 3.8, 1000, -5, 5}},
    {kTH3GMTrackInvQPtResMCHPtEta, {40, 0, 20, 16, 2.2, 3.8, 1000, -5, 5}},
    {kTH3GMTrackXPullPtEta, {40, 0, 20, 16, 2.2, 3.8, 200, -10, 10}},
    {kTH3GMTrackYPullPtEta, {40, 0, 20, 16, 2.2, 3.8, 200, -10, 10}},
    {kTH3GMTrackPhiPullPtEta, {40, 0, 20, 16, 2.2, 3.8, 200, -10, 10}},
    {kTH3GMTrackCotlPullPtEta, {40, 0, 20, 16, 2.2, 3.8, 200, -10, 10}},
    {kTH3GMTrackInvQPtPullPtEta, {40, 0, 20, 16, 2.2, 3.8, 200, -50, 50}},
    {kTH3GMTrackReducedChi2PtEta, {40, 0, 20, 16, 2.2, 3.8, 1000, 0, 100}},
    {kTH3GMCloseMatchPtEtaChi2, {40, 0, 20, 16, 2.2, 3.8, 1000, 0, 100}},
    {kTH3GMCloseMatchPtEtaMatchScore, {40, 0, 20, 16, 2.2, 3.8, 2000, 0, 20.0}},
    {kTH3GMPairablePtEtaZ, {40, 0, 20, 16, 2.2, 3.8, 30, -15, 15}},
    {kTH3GMTrackPtEtaChi2, {40, 0, 20, 16, 2.2, 3.8, 1000, 0, 100}},
    {kTH3GMTrackPtEtaMatchScore, {40, 0, 20, 16, 2.2, 3.8, 2000, 0, 20.0}},
    {kTH3GMTruePtEtaChi2, {40, 0, 20, 16, 2.2, 3.8, 1000, 0, 100}},
    {kTH3GMTruePtEtaMatchScore, {40, 0, 20, 16, 2.2, 3.8, 2000, 0, 20.0}},
    {kTH3GMTruePtEtaMatchScore_MC, {40, 0, 20, 16, 2.2, 3.8, 2000, 0, 20.0}}};

  std::map<int, const char*> TH3XaxisTitles{
    {kTH3GMTrackDeltaXDeltaYEta, R"(\\eta_{MC})"},
    {kTH3GMTrackDeltaXDeltaYPt, R"(p_{t}_{MC})"},
    {kTH3GMTrackDeltaXVertexPtEta, R"(p_{t}_{MC})"},
    {kTH3GMTrackDeltaYVertexPtEta, R"(p_{t}_{MC})"},
    {kTH3GMTrackInvQPtResolutionPtEta, R"(p_{t}_{MC})"},
    {kTH3GMTrackInvQPtResMCHPtEta, R"(p_{t}_{MC})"},
    {kTH3GMTrackXPullPtEta, R"(p_{t}_{MC})"},
    {kTH3GMTrackYPullPtEta, R"(p_{t}_{MC})"},
    {kTH3GMTrackPhiPullPtEta, R"(p_{t}_{MC})"},
    {kTH3GMTrackCotlPullPtEta, R"(p_{t}_{MC})"},
    {kTH3GMTrackInvQPtPullPtEta, R"(p_{t}_{MC})"},
    {kTH3GMTrackReducedChi2PtEta, R"(p_{t}_{MC})"},
    {kTH3GMCloseMatchPtEtaChi2, R"(p_{t}_{Fit})"},
    {kTH3GMCloseMatchPtEtaMatchScore, R"(p_{t}_{Fit})"},
    {kTH3GMPairablePtEtaZ, R"(p_{t}_{MC})"},
    {kTH3GMTrackPtEtaChi2, R"(p_{t}_{Fit})"},
    {kTH3GMTrackPtEtaMatchScore, R"(p_{t}_{Fit})"},
    {kTH3GMTruePtEtaChi2, R"(p_{t}_{Fit})"},
    {kTH3GMTruePtEtaMatchScore, R"(p_{t}_{Fit})"},
    {kTH3GMTruePtEtaMatchScore_MC, R"(p_{t}_{MC})"}};

  std::map<int, const char*> TH3YaxisTitles{
    {kTH3GMTrackDeltaXDeltaYEta, R"(X_{residual \rightarrow vtx} (\mu m))"},
    {kTH3GMTrackDeltaXDeltaYPt, R"(X_{residual \rightarrow vtx} (\mu m))"},
    {kTH3GMTrackDeltaXVertexPtEta, R"(\eta_{MC}v)"},
    {kTH3GMTrackDeltaYVertexPtEta, R"(\eta_{MC})"},
    {kTH3GMTrackInvQPtResolutionPtEta, R"(\eta_{MC})"},
    {kTH3GMTrackInvQPtResMCHPtEta, R"(\eta_{MC})"},
    {kTH3GMTrackXPullPtEta, R"(\eta_{MC})"},
    {kTH3GMTrackYPullPtEta, R"(\eta_{MC})"},
    {kTH3GMTrackPhiPullPtEta, R"(\eta_{MC})"},
    {kTH3GMTrackCotlPullPtEta, R"(\eta_{MC})"},
    {kTH3GMTrackInvQPtPullPtEta, R"(\eta_{MC})"},
    {kTH3GMTrackReducedChi2PtEta, R"(\eta_{MC})"},
    {kTH3GMCloseMatchPtEtaChi2, R"(\eta_{Fit})"},
    {kTH3GMCloseMatchPtEtaMatchScore, R"(\eta_{Fit})"},
    {kTH3GMPairablePtEtaZ, R"(\eta_{MC})"},
    {kTH3GMTrackPtEtaChi2, R"(\eta_{Fit})"},
    {kTH3GMTrackPtEtaMatchScore, R"(\eta_{Fit})"},
    {kTH3GMTruePtEtaChi2, R"(\eta_{Fit})"},
    {kTH3GMTruePtEtaMatchScore, R"(\eta_{Fit})"},
    {kTH3GMTruePtEtaMatchScore_MC, R"(\eta_{MC})"}};

  std::map<int, const char*> TH3ZaxisTitles{
    {kTH3GMTrackDeltaXDeltaYEta, R"(Y_{residual \rightarrow vtx} (\mu m))"},
    {kTH3GMTrackDeltaXDeltaYPt, R"(Y_{residual \rightarrow vtx} (\mu m))"},
    {kTH3GMTrackDeltaXVertexPtEta, R"(X_{residual \rightarrow vtx} (\mu m))"},
    {kTH3GMTrackDeltaYVertexPtEta, R"(Y_{residual \rightarrow vtx} (\mu m))"},
    {kTH3GMTrackInvQPtResolutionPtEta, R"((q/p_{t})_{residual}/(q/p_{t}))"},
    {kTH3GMTrackInvQPtResMCHPtEta, R"((q/p_{t})_{residual}/(q/p_{t}))"},
    {kTH3GMTrackXPullPtEta, R"(\Delta X/\sigma_{X})"},
    {kTH3GMTrackYPullPtEta, R"(\Delta Y/\sigma_{Y})"},
    {kTH3GMTrackPhiPullPtEta, R"(\Delta \phi/\sigma_{\phi})"},
    {kTH3GMTrackCotlPullPtEta, R"(\Delta \cot(\lambda)/\sigma_{cot(\lambda)})"},
    {kTH3GMTrackInvQPtPullPtEta, R"((\Delta q/p_t)/\sigma_{q/p_{t}})"},
    {kTH3GMTrackReducedChi2PtEta, R"(\chi^2/d.f.)"},
    {kTH3GMCloseMatchPtEtaChi2, R"(Match \chi^2)"},
    {kTH3GMCloseMatchPtEtaMatchScore, R"(Matching Score)"},
    {kTH3GMPairablePtEtaZ, R"(z_{vtx})"},
    {kTH3GMTrackPtEtaChi2, R"(Match \chi^2)"},
    {kTH3GMTrackPtEtaMatchScore, R"(Matching Score)"},
    {kTH3GMTruePtEtaChi2, R"(Match \chi^2)"},
    {kTH3GMTruePtEtaMatchScore, R"(Matching Score)"},
    {kTH3GMTruePtEtaMatchScore_MC, R"(Matching Score)"}};

  enum TH3SlicedCodes {
    kDeltaXVertexVsEta,
    kDeltaXVertexVsPt,
    kDeltaYVertexVsEta,
    kDeltaYVertexVsPt,
    kXPullVsEta,
    kXPullVsPt,
    kYPullVsEta,
    kYPullVsPt,
    kInvQPtResVsEta,
    kInvQPtResVsPt,
    kInvQPtResMCHVsEta,
    kInvQPtResMCHVsPt,
    kPhiPullVsEta,
    kPhiPullVsPt,
    kCotlPullVsEta,
    kCotlPullVsPt,
    kInvQPtPullVsEta,
    kInvQPtPullVsPt,
    kNSlicedTH3
  };

  std::map<int, const char*> TH3SlicedNames{
    {kDeltaXVertexVsEta, "DeltaXVertexVsEta"},
    {kDeltaXVertexVsPt, "DeltaXVertexVsPt"},
    {kDeltaYVertexVsEta, "DeltaYVertexVsEta"},
    {kDeltaYVertexVsPt, "DeltaYVertexVsPt"},
    {kXPullVsEta, "XPullVsEta"},
    {kXPullVsPt, "XPullVsPt"},
    {kYPullVsEta, "YPullVsEta"},
    {kYPullVsPt, "YPullVsPt"},
    {kInvQPtResVsEta, "InvQPtResVsEta"},
    {kInvQPtResVsPt, "InvQPtResVsPt"},
    {kInvQPtResMCHVsEta, "InvQPtResMCHVsEta"},
    {kInvQPtResMCHVsPt, "InvQPtResMCHVsPt"},
    {kPhiPullVsEta, "PhiPullVsEta"},
    {kPhiPullVsPt, "PhiPullVsPt"},
    {kCotlPullVsEta, "CotlPullVsEta"},
    {kCotlPullVsPt, "CotlPullVsPt"},
    {kInvQPtPullVsEta, "InvQPtPullVsEta"},
    {kInvQPtPullVsPt, "InvQPtPullVsPt"}};

  std::map<int, int> TH3SlicedMap{
    {kDeltaXVertexVsEta, kTH3GMTrackDeltaXVertexPtEta},
    {kDeltaXVertexVsPt, kTH3GMTrackDeltaXVertexPtEta},
    {kDeltaYVertexVsEta, kTH3GMTrackDeltaYVertexPtEta},
    {kDeltaYVertexVsPt, kTH3GMTrackDeltaYVertexPtEta},
    {kXPullVsEta, kTH3GMTrackXPullPtEta},
    {kXPullVsPt, kTH3GMTrackXPullPtEta},
    {kYPullVsEta, kTH3GMTrackYPullPtEta},
    {kYPullVsPt, kTH3GMTrackYPullPtEta},
    {kInvQPtResVsEta, kTH3GMTrackInvQPtResolutionPtEta},
    {kInvQPtResVsPt, kTH3GMTrackInvQPtResolutionPtEta},
    {kInvQPtResMCHVsEta, kTH3GMTrackInvQPtResMCHPtEta},
    {kInvQPtResMCHVsPt, kTH3GMTrackInvQPtResMCHPtEta},
    {kPhiPullVsEta, kTH3GMTrackPhiPullPtEta},
    {kPhiPullVsPt, kTH3GMTrackPhiPullPtEta},
    {kCotlPullVsEta, kTH3GMTrackCotlPullPtEta},
    {kCotlPullVsPt, kTH3GMTrackCotlPullPtEta},
    {kInvQPtPullVsEta, kTH3GMTrackInvQPtPullPtEta},
    {kInvQPtPullVsPt, kTH3GMTrackInvQPtPullPtEta}};

  std::array<std::unique_ptr<TH3F>, kNTH3Histos> mTH3Histos;
  std::array<TCanvas*, kNSlicedTH3> mSlicedCanvas;
  void TH3Slicer(TCanvas* canvas, std::unique_ptr<TH3F>& histo3D, std::vector<float> list, double window, int iPar, float marker_size = 1.5);

  std::unordered_map<o2::MCCompLabel, bool> mPairables;
  std::vector<std::vector<std::vector<int>>> mTrueTracksMap;                 // Maps srcIDs and eventIDs to true reco tracks
  std::vector<std::vector<std::vector<o2::MCCompLabel>>> mPairableTracksMap; // Maps srcIDs and eventIDs to pairable tracks

  enum GMAssesmentCanvases {
    kPurityPtOuter,
    kPurityPtInner,
    kPairingEffPtOuter,
    kPairingEffPtInner,
    kPurityVsEfficiency,
    kTruePairingEffPtOuter,
    kTruePairingEffPtInner,
    kPurityVsTrueEfficiency,
    kNGMAssesmentCanvases
  };

  std::map<int, const char*> GMAssesmentNames{
    {kPurityPtOuter, "PurityPtOuter"},
    {kPurityPtInner, "PurityPtInner"},
    {kPairingEffPtOuter, "PairingEffPtOuter"},
    {kPairingEffPtInner, "PairingEffPtInner"},
    {kTruePairingEffPtOuter, "TruePairingEffPtOuter"},
    {kTruePairingEffPtInner, "TruePairingEffPtInner"},
    {kPurityVsEfficiency, "PurityVsEfficiency"},
    {kPurityVsTrueEfficiency, "PurityVsTrueEfficiency"}};

  std::array<TCanvas*, kNGMAssesmentCanvases> mAssessmentCanvas;

  static constexpr std::array<short, 7> sMinNClustersList = {4, 5, 6, 7, 8, 9, 10};
  uint32_t mRefOrbit = 0; // Reference orbit used in relative time calculation
  float mBz = 0;
  bool mMIDFilterEnabled = true;
  bool mFinalizeAnalysis = false;
  float mFinalizeMinCut = 0.f;
  float mFinalizeMaxCut = 15.f;
  int mNFinalizeSteps = 15;

  ClassDefNV(GloFwdAssessment, 1);
};

} // namespace globaltracking
} // namespace o2

#endif
