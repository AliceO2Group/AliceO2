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

namespace o2
{

namespace mft
{

enum mMFTTrackTypes { kReco,
                      kGen,
                      kTrackable,
                      kRecoTrue,
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

  bool init();
  void runASyncQC(o2::framework::ProcessingContext& ctx);
  void processTrackables();
  void processGeneratedTracks();
  void processRecoAndTrueTracks();
  void addMCParticletoHistos(const MCTrack* mcTr, const int TrackType, const o2::dataformats::MCEventHeader& evH);
  void finalize();
  void reset();

  void getHistos(TObjArray& objar);
  void deleteHistograms();
  void setBz(float bz) { mBz = bz; }

  double orbitToSeconds(uint32_t orbit, uint32_t refOrbit)
  {
    return (orbit - refOrbit) * o2::constants::lhc::LHCOrbitNS / 1E9;
  }

 private:
  gsl::span<const o2::mft::TrackMFT> mMFTTracks;
  gsl::span<const o2::itsmft::ROFRecord> mMFTTracksROF;
  gsl::span<const o2::itsmft::CompClusterExt> mMFTClusters;
  gsl::span<const o2::itsmft::ROFRecord> mMFTClustersROF;

  // MC Labels
  bool mUseMC = false;

  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>> mMFTClusterLabels;
  gsl::span<const o2::MCCompLabel> mMFTTrackLabels;

  o2::steer::MCKinematicsReader mcReader; // reader of MC information

  // Histos for reconstructed tracks
  std::unique_ptr<TH1F> mTrackNumberOfClusters = nullptr;
  std::unique_ptr<TH1F> mCATrackNumberOfClusters = nullptr;
  std::unique_ptr<TH1F> mLTFTrackNumberOfClusters = nullptr;
  std::unique_ptr<TH1F> mTrackOnvQPt = nullptr;
  std::unique_ptr<TH1F> mTrackChi2 = nullptr;
  std::unique_ptr<TH1F> mTrackCharge = nullptr;
  std::unique_ptr<TH1F> mTrackPhi = nullptr;
  std::unique_ptr<TH1F> mPositiveTrackPhi = nullptr;
  std::unique_ptr<TH1F> mNegativeTrackPhi = nullptr;
  std::unique_ptr<TH1F> mTrackEta = nullptr;
  std::array<std::unique_ptr<TH1F>, 7> mTrackEtaNCls = {nullptr};
  std::array<std::unique_ptr<TH1F>, 7> mTrackPhiNCls = {nullptr};
  std::array<std::unique_ptr<TH2F>, 7> mTrackXYNCls = {nullptr};
  std::array<std::unique_ptr<TH2F>, 7> mTrackEtaPhiNCls = {nullptr};
  std::unique_ptr<TH1F> mCATrackEta = nullptr;
  std::unique_ptr<TH1F> mLTFTrackEta = nullptr;
  std::unique_ptr<TH1F> mTrackTanl = nullptr;

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
                                                "RecoTrue"};

  std::unique_ptr<TH2F> mHistPhiRecVsPhiGen = nullptr;
  std::unique_ptr<TH2F> mHistEtaRecVsEtaGen = nullptr;

  std::array<std::unique_ptr<TH2F>, kNumberOfTrackTypes> mHistPhiVsEta;
  std::array<std::unique_ptr<TH2F>, kNumberOfTrackTypes> mHistPtVsEta;
  std::array<std::unique_ptr<TH2F>, kNumberOfTrackTypes> mHistPhiVsPt;
  std::array<std::unique_ptr<TH2F>, kNumberOfTrackTypes> mHistZvtxVsEta;
  std::array<std::unique_ptr<TH2F>, kNumberOfTrackTypes> mHistRVsZ;

  std::unordered_map<o2::MCCompLabel, bool> mMFTTrackables;

  static constexpr std::array<short, 7> sMinNClustersList = {4, 5, 6, 7, 8, 9, 10};
  uint32_t mRefOrbit = 0; // Reference orbit used in relative time calculation
  float mBz = 0;

  o2::itsmft::ChipMappingMFT mMFTChipMapper;

  ClassDefNV(MFTAssessment, 1);
};

} // namespace mft
} // namespace o2

#endif
