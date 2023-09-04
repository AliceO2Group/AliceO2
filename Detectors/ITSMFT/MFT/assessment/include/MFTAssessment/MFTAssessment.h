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

/// \file MFTAssessment.h
/// \brief Class to perform assessment of MFT
/// \author rafael.pezzi at cern.ch

#ifndef ALICEO2_MFT_ASSESSMENT
#define ALICEO2_MFT_ASSESSMENT

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
#include <DataFormatsITSMFT/ROFRecord.h>
#include <DataFormatsITSMFT/CompCluster.h>
#include "ITSMFTReconstruction/ChipMappingMFT.h"
#include "Steer/MCKinematicsReader.h"
#include <unordered_map>
#include <vector>
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "MFTTracking/IOUtils.h"
#include "ReconstructionDataFormats/BaseCluster.h"
#include "ITSMFTReconstruction/ClustererParam.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"

namespace o2
{

namespace mft
{

enum mMFTTrackTypes { kReco,
                      kGen,
                      kTrackable,
                      kRecoTrue,
                      kRecoFake,
                      kRecoTrueMC,
                      kNumberOfTrackTypes };

using ClusterLabelsType = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
using TrackLabelsType = std::vector<o2::MCCompLabel>;
using MCTrack = o2::MCTrackT<float>;

class MFTAssessment
{
 public:
  MFTAssessment() = delete;
  MFTAssessment(bool useMC) : mUseMC(useMC){};
  ~MFTAssessment() = default;

  void init(bool finalizeAnalysis);
  void setRefOrbit(uint32_t orbit) { mRefOrbit = orbit; }
  void createHistos();
  void runASyncQC(o2::framework::ProcessingContext& ctx);
  void processTrackables();
  void processGeneratedTracks();
  void processRecoTracks();
  void processTrueAndFakeTracks();
  void addMCParticletoHistos(const MCTrack* mcTr, const int TrackType, const o2::dataformats::MCEventHeader& evH);
  void reset();
  void fillTrueRecoTracksMap()
  {
    mTrueTracksMap.resize(mMCReader.getNSources());
    auto src = 0;
    for (auto& map : mTrueTracksMap) {
      map.resize(mMCReader.getNEvents(src++));
    }
    auto id = 0;
    for (const auto& trackLabel : mMFTTrackLabels) {
      if (trackLabel.isCorrect()) {
        mTrueTracksMap[trackLabel.getSourceID()][trackLabel.getEventID()].push_back(id);
      } else {
        mFakeTracksVec.push_back(id);
      }
      id++;
    }
  }

  bool loadHistos();
  void finalizeAnalysis();

  void getHistos(TObjArray& objar);
  void deleteHistograms();
  void setBz(float bz) { mBz = bz; }
  void setClusterDictionary(const o2::itsmft::TopologyDictionary* d) { mDictionary = d; }
  double orbitToSeconds(uint32_t orbit, uint32_t refOrbit)
  {
    return (orbit - refOrbit) * o2::constants::lhc::LHCOrbitNS / 1E9;
  }

 private:
  const o2::itsmft::TopologyDictionary* mDictionary = nullptr; // cluster patterns dictionary

  gsl::span<const o2::mft::TrackMFT> mMFTTracks;
  gsl::span<const o2::itsmft::ROFRecord> mMFTTracksROF;
  gsl::span<const int> mMFTTrackClusIdx;
  gsl::span<const o2::itsmft::CompClusterExt> mMFTClusters;
  gsl::span<const o2::itsmft::ROFRecord> mMFTClustersROF;
  gsl::span<const unsigned char> mMFTClusterPatterns;
  gsl::span<const unsigned char>::iterator pattIt;
  std::vector<o2::BaseCluster<float>> mMFTClustersGlobal;

  std::array<bool, 936> mUnusedChips; // 936 chipIDs in total
  int mNumberTFs = 0;
  int mLastTrackType;

  // MC Labels
  bool mUseMC = false;

  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>> mMFTClusterLabels;
  gsl::span<const o2::MCCompLabel> mMFTTrackLabels;

  o2::steer::MCKinematicsReader mMCReader; // reader of MC information

  // Histos for reconstructed tracks
  std::unique_ptr<TH1F> mTrackNumberOfClusters = nullptr;
  std::unique_ptr<TH1F> mCATrackNumberOfClusters = nullptr;
  std::unique_ptr<TH1F> mLTFTrackNumberOfClusters = nullptr;
  std::unique_ptr<TH1F> mTrackInvQPt = nullptr;
  std::unique_ptr<TH1F> mTrackChi2 = nullptr;
  std::unique_ptr<TH1F> mTrackCharge = nullptr;
  std::unique_ptr<TH1F> mTrackPhi = nullptr;
  std::unique_ptr<TH1F> mPositiveTrackPhi = nullptr;
  std::unique_ptr<TH1F> mNegativeTrackPhi = nullptr;
  std::unique_ptr<TH1F> mTrackEta = nullptr;

  std::unique_ptr<TH1F> mMFTClsZ = nullptr;
  std::unique_ptr<TH1F> mMFTClsOfTracksZ = nullptr;

  std::array<std::unique_ptr<TH2F>, 10> mMFTClsXYinLayer = {nullptr};
  std::array<std::unique_ptr<TH1F>, 10> mMFTClsRinLayer = {nullptr};
  std::array<std::unique_ptr<TH2F>, 10> mMFTClsOfTracksXYinLayer = {nullptr};
  std::array<std::unique_ptr<TH2F>, 5> mMFTClsXYRedundantInDisk = {nullptr};

  std::array<std::unique_ptr<TH1F>, 7> mTrackEtaNCls = {nullptr};
  std::array<std::unique_ptr<TH1F>, 7> mTrackPhiNCls = {nullptr};
  std::array<std::unique_ptr<TH2F>, 7> mTrackXYNCls = {nullptr};
  std::array<std::unique_ptr<TH2F>, 7> mTrackEtaPhiNCls = {nullptr};
  std::unique_ptr<TH1F> mCATrackEta = nullptr;
  std::unique_ptr<TH1F> mLTFTrackEta = nullptr;
  std::unique_ptr<TH1F> mTrackCotl = nullptr;

  std::unique_ptr<TH1F> mTrackROFNEntries = nullptr;
  std::unique_ptr<TH1F> mClusterROFNEntries = nullptr;
  std::unique_ptr<TH1F> mTracksBC = nullptr;

  std::unique_ptr<TH1F> mNOfTracksTime = nullptr;
  std::unique_ptr<TH1F> mNOfClustersTime = nullptr;

  std::unique_ptr<TH1F> mClusterSensorIndex = nullptr;
  std::unique_ptr<TH1F> mClusterPatternIndex = nullptr;

  // Histos and data for MC analysis
  std::vector<std::string> mNameOfTrackTypes = {"Rec",
                                                "Gen",
                                                "Trackable",
                                                "RecoTrue",
                                                "RecoFake",
                                                "RecoTrueMC"};

  std::unique_ptr<TH2F> mHistPhiRecVsPhiGen = nullptr;
  std::unique_ptr<TH2F> mHistEtaRecVsEtaGen = nullptr;

  std::array<std::unique_ptr<TH2F>, kNumberOfTrackTypes> mHistPhiVsEta;
  std::array<std::unique_ptr<TH2F>, kNumberOfTrackTypes> mHistPtVsEta;
  std::array<std::unique_ptr<TH2F>, kNumberOfTrackTypes> mHistPhiVsPt;
  std::array<std::unique_ptr<TH2F>, kNumberOfTrackTypes> mHistZvtxVsEta;
  std::array<std::unique_ptr<TH2F>, kNumberOfTrackTypes> mHistRVsZ;
  std::array<std::unique_ptr<TH1F>, kNumberOfTrackTypes> mHistIsPrimary;
  std::array<std::unique_ptr<TH1F>, kNumberOfTrackTypes> mHistTrackChi2;

  // Histos for reconstruction assessment

  std::unique_ptr<TEfficiency> mChargeMatchEff = nullptr;
  std::unique_ptr<TH2F> mHistVxtOffsetProjection = nullptr;

  enum TH3HistosCodes {
    kTH3TrackDeltaXDeltaYEta,
    kTH3TrackDeltaXDeltaYPt,
    kTH3TrackDeltaXVertexPtEta,
    kTH3TrackDeltaYVertexPtEta,
    kTH3TrackInvQPtResolutionPtEta,
    kTH3TrackInvQPtResSeedPtEta,
    kTH3TrackXPullPtEta,
    kTH3TrackYPullPtEta,
    kTH3TrackPhiPullPtEta,
    kTH3TrackCotlPullPtEta,
    kTH3TrackInvQPtPullPtEta,
    kTH3TrackReducedChi2PtEta,
    kNTH3Histos
  };

  std::map<int, const char*> TH3Names{
    {kTH3TrackDeltaXDeltaYEta, "TH3TrackDeltaXDeltaYEta"},
    {kTH3TrackDeltaXDeltaYPt, "TH3TrackDeltaXDeltaYPt"},
    {kTH3TrackDeltaXVertexPtEta, "TH3TrackDeltaXVertexPtEta"},
    {kTH3TrackDeltaYVertexPtEta, "TH3TrackDeltaYVertexPtEta"},
    {kTH3TrackInvQPtResolutionPtEta, "TH3TrackInvQPtResolutionPtEta"},
    {kTH3TrackInvQPtResSeedPtEta, "TH3TrackInvQPtResSeedPtEta"},
    {kTH3TrackXPullPtEta, "TH3TrackXPullPtEta"},
    {kTH3TrackYPullPtEta, "TH3TrackYPullPtEta"},
    {kTH3TrackPhiPullPtEta, "TH3TrackPhiPullPtEta"},
    {kTH3TrackCotlPullPtEta, "TH3TrackCotlPullPtEta"},
    {kTH3TrackInvQPtPullPtEta, "TH3TrackInvQPtPullPtEta"},
    {kTH3TrackReducedChi2PtEta, "TH3TrackReducedChi2PtEta"}};

  std::map<int, const char*> TH3Titles{
    {kTH3TrackDeltaXDeltaYEta, "TH3TrackDeltaXDeltaYEta"},
    {kTH3TrackDeltaXDeltaYPt, "TH3TrackDeltaXDeltaYPt"},
    {kTH3TrackDeltaXVertexPtEta, "TH3TrackDeltaXVertexPtEta"},
    {kTH3TrackDeltaYVertexPtEta, "TH3TrackDeltaYVertexPtEta"},
    {kTH3TrackInvQPtResolutionPtEta, "TH3TrackInvQPtResolutionPtEta"},
    {kTH3TrackInvQPtResSeedPtEta, "TH3TrackInvQPtResSeedPtEta"},
    {kTH3TrackXPullPtEta, "TH3TrackXPullPtEta"},
    {kTH3TrackYPullPtEta, "TH3TrackYPullPtEta"},
    {kTH3TrackPhiPullPtEta, "TH3TrackPhiPullPtEta"},
    {kTH3TrackCotlPullPtEta, "TH3TrackCotlPullPtEta"},
    {kTH3TrackInvQPtPullPtEta, "TH3TrackInvQPtPullPtEta"},
    {kTH3TrackReducedChi2PtEta, "TH3TrackReducedChi2PtEta"}};

  std::map<int, std::array<double, 9>> TH3Binning{
    {kTH3TrackDeltaXDeltaYEta, {16, -3.8, -2.2, 1000, -1000, 1000, 1000, -1000, 1000}},
    {kTH3TrackDeltaXDeltaYPt, {100, 0, 20, 1000, -1000, 1000, 1000, -1000, 1000}},
    {kTH3TrackDeltaYVertexPtEta, {100, 0, 20, 16, -3.8, -2.2, 1000, -1000, 1000}},
    {kTH3TrackDeltaXVertexPtEta, {100, 0, 20, 16, -3.8, -2.2, 1000, -1000, 1000}},
    {kTH3TrackInvQPtResolutionPtEta, {100, 0, 20, 16, -3.8, -2.2, 1000, -50, 50}},
    {kTH3TrackInvQPtResSeedPtEta, {100, 0, 20, 16, -3.8, -2.2, 1000, -50, 50}},
    {kTH3TrackXPullPtEta, {100, 0, 20, 16, -3.8, -2.2, 200, -10, 10}},
    {kTH3TrackYPullPtEta, {100, 0, 20, 16, -3.8, -2.2, 200, -10, 10}},
    {kTH3TrackPhiPullPtEta, {100, 0, 20, 16, -3.8, -2.2, 200, -10, 10}},
    {kTH3TrackCotlPullPtEta, {100, 0, 20, 16, -3.8, -2.2, 200, -10, 10}},
    {kTH3TrackInvQPtPullPtEta, {100, 0, 20, 16, -3.8, -2.2, 1000, -15, 15}},
    {kTH3TrackReducedChi2PtEta, {100, 0, 20, 16, -3.8, -2.2, 1000, 0, 100}}};

  std::map<int, const char*> TH3XaxisTitles{
    {kTH3TrackDeltaXDeltaYEta, R"(\\eta)"},
    {kTH3TrackDeltaXDeltaYPt, R"(p_{t})"},
    {kTH3TrackDeltaXVertexPtEta, R"(p_{t})"},
    {kTH3TrackDeltaYVertexPtEta, R"(p_{t})"},
    {kTH3TrackInvQPtResolutionPtEta, R"(p_{t})"},
    {kTH3TrackInvQPtResSeedPtEta, R"(p_{t})"},
    {kTH3TrackXPullPtEta, R"(p_{t})"},
    {kTH3TrackYPullPtEta, R"(p_{t})"},
    {kTH3TrackPhiPullPtEta, R"(p_{t})"},
    {kTH3TrackCotlPullPtEta, R"(p_{t})"},
    {kTH3TrackInvQPtPullPtEta, R"(p_{t})"},
    {kTH3TrackReducedChi2PtEta, R"(p_{t})"}};

  std::map<int, const char*> TH3YaxisTitles{
    {kTH3TrackDeltaXDeltaYEta, R"(X_{residual \rightarrow vtx} (\mu m))"},
    {kTH3TrackDeltaXDeltaYPt, R"(X_{residual \rightarrow vtx} (\mu m))"},
    {kTH3TrackDeltaXVertexPtEta, R"(\eta)"},
    {kTH3TrackDeltaYVertexPtEta, R"(\eta)"},
    {kTH3TrackInvQPtResolutionPtEta, R"(\eta)"},
    {kTH3TrackInvQPtResSeedPtEta, R"(\eta)"},
    {kTH3TrackXPullPtEta, R"(\eta)"},
    {kTH3TrackYPullPtEta, R"(\eta)"},
    {kTH3TrackPhiPullPtEta, R"(\eta)"},
    {kTH3TrackCotlPullPtEta, R"(\eta)"},
    {kTH3TrackInvQPtPullPtEta, R"(\eta)"},
    {kTH3TrackReducedChi2PtEta, R"(\eta)"}};

  std::map<int, const char*> TH3ZaxisTitles{
    {kTH3TrackDeltaXDeltaYEta, R"(Y_{residual \rightarrow vtx} (\mu m))"},
    {kTH3TrackDeltaXDeltaYPt, R"(Y_{residual \rightarrow vtx} (\mu m))"},
    {kTH3TrackDeltaXVertexPtEta, R"(X_{residual \rightarrow vtx} (\mu m))"},
    {kTH3TrackDeltaYVertexPtEta, R"(Y_{residual \rightarrow vtx} (\mu m))"},
    {kTH3TrackInvQPtResolutionPtEta, R"((q/p_{t})_{residual}/(q/p_{t}))"},
    {kTH3TrackInvQPtResSeedPtEta, R"((q/p_{t})_{residual}/(q/p_{t}))"},
    {kTH3TrackXPullPtEta, R"(\Delta X/\sigma_{X})"},
    {kTH3TrackYPullPtEta, R"(\Delta Y/\sigma_{Y})"},
    {kTH3TrackPhiPullPtEta, R"(\Delta \phi/\sigma_{\phi})"},
    {kTH3TrackCotlPullPtEta, R"(\Delta \cot(\lambda)/\sigma_{cot(\lambda)})"},
    {kTH3TrackInvQPtPullPtEta, R"((\Delta q/p_t)/\sigma_{q/p_{t}})"},
    {kTH3TrackReducedChi2PtEta, R"(\chi^2/d.f.)"}};

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
    kInvQPtResSeedVsEta,
    kInvQPtResSeedVsPt,
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
    {kInvQPtResSeedVsEta, "InvQPtResSeedVsEta"},
    {kInvQPtResSeedVsPt, "InvQPtResSeedVsPt"},
    {kPhiPullVsEta, "PhiPullVsEta"},
    {kPhiPullVsPt, "PhiPullVsPt"},
    {kCotlPullVsEta, "CotlPullVsEta"},
    {kCotlPullVsPt, "CotlPullVsPt"},
    {kInvQPtPullVsEta, "InvQPtPullVsEta"},
    {kInvQPtPullVsPt, "InvQPtPullVsPt"}};

  std::map<int, int> TH3SlicedMap{
    {kDeltaXVertexVsEta, kTH3TrackDeltaXVertexPtEta},
    {kDeltaXVertexVsPt, kTH3TrackDeltaXVertexPtEta},
    {kDeltaYVertexVsEta, kTH3TrackDeltaYVertexPtEta},
    {kDeltaYVertexVsPt, kTH3TrackDeltaYVertexPtEta},
    {kXPullVsEta, kTH3TrackXPullPtEta},
    {kXPullVsPt, kTH3TrackXPullPtEta},
    {kYPullVsEta, kTH3TrackYPullPtEta},
    {kYPullVsPt, kTH3TrackYPullPtEta},
    {kInvQPtResVsEta, kTH3TrackInvQPtResolutionPtEta},
    {kInvQPtResVsPt, kTH3TrackInvQPtResolutionPtEta},
    {kInvQPtResSeedVsEta, kTH3TrackInvQPtResSeedPtEta},
    {kInvQPtResSeedVsPt, kTH3TrackInvQPtResSeedPtEta},
    {kPhiPullVsEta, kTH3TrackPhiPullPtEta},
    {kPhiPullVsPt, kTH3TrackPhiPullPtEta},
    {kCotlPullVsEta, kTH3TrackCotlPullPtEta},
    {kCotlPullVsPt, kTH3TrackCotlPullPtEta},
    {kInvQPtPullVsEta, kTH3TrackInvQPtPullPtEta},
    {kInvQPtPullVsPt, kTH3TrackInvQPtPullPtEta}};

  std::array<std::unique_ptr<TH3F>, kNTH3Histos> mTH3Histos;
  std::array<TCanvas*, kNSlicedTH3> mSlicedCanvas;
  void TH3Slicer(TCanvas* canvas, std::unique_ptr<TH3F>& histo3D, std::vector<float> list, double window, int iPar, float marker_size = 1.5);

  std::unordered_map<o2::MCCompLabel, bool> mMFTTrackables;
  std::vector<std::vector<std::vector<int>>> mTrueTracksMap;                  // Maps srcIDs and eventIDs to true reco tracks
  std::vector<int> mFakeTracksVec;                                            // IDs of fake MFT tracks
  std::vector<std::vector<std::vector<o2::MCCompLabel>>> mTrackableTracksMap; // Maps srcIDs and eventIDs to trackable tracks

  static constexpr std::array<short, 7> sMinNClustersList = {4, 5, 6, 7, 8, 9, 10};
  uint32_t mRefOrbit = 0; // Reference orbit used in relative time calculation
  float mBz = 0;
  bool mFinalizeAnalysis = false;

  o2::itsmft::ChipMappingMFT mMFTChipMapper;

  ClassDefNV(MFTAssessment, 1);
};

} // namespace mft
} // namespace o2

#endif
