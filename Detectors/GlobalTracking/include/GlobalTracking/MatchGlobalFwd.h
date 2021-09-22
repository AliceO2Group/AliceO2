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

/// \file MatchGlobalFwd.h
/// \brief Class to perform MCH MFT matching
/// \author rafael.pezzi@cern.ch

#ifndef ALICEO2_GLOBTRACKING_MATCHGLOBALFWD_
#define ALICEO2_GLOBTRACKING_MATCHGLOBALFWD_

#include <Rtypes.h>
#include <array>
#include <vector>
#include <string>
#include <gsl/span>
#include <TStopwatch.h>
#include "CommonConstants/LHCConstants.h"
#include "CommonDataFormat/BunchFilling.h"
#include "ITSMFTReconstruction/ChipMappingMFT.h"
#include "MCHTracking/TrackExtrap.h"
#include "MCHTracking/TrackParam.h"
#include "MFTTracking/IOUtils.h"
#include "MFTBase/Constants.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "ReconstructionDataFormats/GlobalFwdTrack.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DetectorsBase/GeometryManager.h"
#include "TGeoManager.h"

namespace o2
{

namespace mft
{
class TrackMFT;
}

namespace mch
{
class TrackMCH;
}

namespace globaltracking
{

///< MFT track outward parameters propagated to reference Z,
///<  with time bracket and index of original track in the
///<  currently loaded MFT reco output
struct TrackLocMFT : public o2::track::TrackParCovFwd {
  o2::math_utils::Bracketf_t tBracket; ///< bracketing time in \mus
  int roFrame = -1;                    ///< MFT readout frame assigned to this track

  ClassDefNV(TrackLocMFT, 0);
};

///< MCH track uncorrected parameters propagated to reference Z,
///<  with time bracket and index of original track in the
///<  currently loaded MCH reco output
struct TrackLocMCH : public o2::dataformats::GlobalFwdTrack {
  o2::math_utils::Bracketf_t tBracket; ///< bracketing time in \mus
  ClassDefNV(TrackLocMCH, 0);
};

class MatchGlobalFwd
{
 public:
  using MFTCluster = o2::BaseCluster<float>;
  using BracketF = o2::math_utils::Bracket<float>;
  using SMatrix55Std = ROOT::Math::SMatrix<double, 5>;
  using SMatrix55Sym = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;

  using SVector2 = ROOT::Math::SVector<double, 2>;
  using SVector4 = ROOT::Math::SVector<double, 4>;
  using SVector5 = ROOT::Math::SVector<double, 5>;

  using SMatrix44 = ROOT::Math::SMatrix<double, 4>;
  using SMatrix45 = ROOT::Math::SMatrix<double, 4, 5>;
  using SMatrix54 = ROOT::Math::SMatrix<double, 5, 4>;
  using SMatrix22 = ROOT::Math::SMatrix<double, 2>;
  using SMatrix25 = ROOT::Math::SMatrix<double, 2, 5>;
  using SMatrix52 = ROOT::Math::SMatrix<double, 5, 2>;

  MatchGlobalFwd() = default;
  ~MatchGlobalFwd() = default;

  void run(const o2::globaltracking::RecoContainer& inp);
  void init(std::string matchFcn, std::string cutFcn);
  void finalize();
  void clear();

  void setMFTDictionary(const o2::itsmft::TopologyDictionary* d) { mMFTDict = d; }
  void setMatchingPlaneZ(float z) { mMatchingPlaneZ = z; };

  ///< set Bunch filling and init helpers for validation by BCs
  void setBunchFilling(const o2::BunchFilling& bf);
  ///< MFT readout mode
  void setMFTTriggered(bool v) { mMFTTriggered = v; }
  bool isMFTTriggered() const { return mMFTTriggered; }

  void setMCTruthOn(bool v) { mMCTruthON = v; }
  ///< set MFT ROFrame duration in microseconds
  void setMFTROFrameLengthMUS(float fums);
  ///< set MFT ROFrame duration in BC (continuous mode only)
  void setMFTROFrameLengthInBC(int nbc);
  const std::vector<o2::dataformats::GlobalFwdTrack>& getMatchedFwdTracks() const { return mMatchedTracks; }
  const std::vector<o2::MCCompLabel>& getMatchLabels() const { return mMatchLabels; }

 private:
  void updateTimeDependentParams();

  bool prepareMCHData();
  bool prepareMFTData();

  void doMatching();
  ///< Matches MFT tracks in one MFT ROFrame with all MCH tracks in the overlapping MCH ROFrames
  void ROFMatch(int MFTROFId, int firstMCHROFId, int lastMCHROFId);
  void fitTracks();                                          // Fit all matched tracks
  void fitGlobalMuonTrack(o2::dataformats::GlobalFwdTrack&); // Kalman filter fit global Forward track by attaching MFT clusters
  bool computeCluster(o2::dataformats::GlobalFwdTrack& track, const MFTCluster& cluster, int& startingLayerID);

  template <typename T>
  bool propagateToNextClusterWithMCS(T& track, double z, int& startingLayerID, const int& newLayerID)
  {
    // Propagate track to the next cluster z position, adding angular MCS effects at the center of
    // each disk crossed by the track. This method is valid only for track propagation between
    // clusters at MFT layers positions. The startingLayerID is updated.

    if (startingLayerID == newLayerID) { // Same layer, nothing to do.
      LOG(DEBUG) << " => Propagate to next cluster with MCS : startingLayerID = " << startingLayerID << " = > "
                 << " newLayerID = " << newLayerID << " (NLayers = " << std::abs(newLayerID - startingLayerID)
                 << ") ; track.getZ() = " << track.getZ() << " => "
                 << "destination cluster z = " << z << " ; => Same layer: no MCS effects.";

      if (z != track.getZ()) {
        track.propagateToZ(z, mBz);
      }
      return true;
    }

    using o2::mft::constants::LayerZPosition;
    auto startingZ = track.getZ();

    //https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
    auto signum = [](auto a) {
      return (0 < a) - (a < 0);
    };
    int direction = signum(newLayerID - startingLayerID); // takes values +1, 0, -1
    auto currentLayer = startingLayerID;

    LOG(DEBUG) << " => Propagate to next cluster with MCS : startingLayerID = " << startingLayerID << " = > "
               << " newLayerID = " << newLayerID << " (NLayers = " << std::abs(newLayerID - startingLayerID)
               << ") ; track.getZ() = " << track.getZ() << " => "
               << "destination cluster z = " << z << " ; ";

    // Number of disks crossed by this track segment
    while (currentLayer != newLayerID) {
      auto nextlayer = currentLayer + direction;
      auto nextZ = LayerZPosition[nextlayer];

      int NDisksMS;
      if (nextZ - track.getZ() > 0) {
        NDisksMS = (currentLayer % 2 == 0) ? (currentLayer - nextlayer) / 2 : (currentLayer - nextlayer + 1) / 2;
      } else {
        NDisksMS = (currentLayer % 2 == 0) ? (nextlayer - currentLayer + 1) / 2 : (nextlayer - currentLayer) / 2;
      }

      LOG(DEBUG) << "currentLayer = " << currentLayer << " ; "
                 << "nextlayer = " << nextlayer << " ; "
                 << "track.getZ() = " << track.getZ() << " ; "
                 << "nextZ = " << nextZ << " ; "
                 << "NDisksMS = " << NDisksMS;

      if ((NDisksMS * mMFTDiskThicknessInX0) != 0) {
        track.addMCSEffect(NDisksMS * mMFTDiskThicknessInX0);
        LOG(DEBUG) << "Track covariances after MCS effects:";
        LOG(DEBUG) << track.getCovariances() << std::endl;
      }

      LOG(DEBUG) << "  BeforeExtrap: X = " << track.getX() << " Y = " << track.getY() << " Z = " << track.getZ() << " Tgl = " << track.getTanl() << "  Phi = " << track.getPhi() << " pz = " << track.getPz() << " q/pt = " << track.getInvQPt() << std::endl;

      track.propagateToZ(nextZ, mBz);

      currentLayer = nextlayer;
    }
    if (z != track.getZ()) {
      track.propagateToZ(z, mBz);
    }
    startingLayerID = newLayerID;
    return true;
  }

  void configMatching(const std::string& matchingFcn, const std::string& cutFcn);
  double matchingEval(const TrackLocMCH& mchTrack, const TrackLocMFT& mftTrack);
  bool matchingCut(const TrackLocMCH&, const TrackLocMFT&);
  double (MatchGlobalFwd::*mMatchFunc)(const TrackLocMCH& mchTrack, const TrackLocMFT& mftTrack);
  void setMatchingFunction(double (MatchGlobalFwd::*func)(const TrackLocMCH&, const TrackLocMFT&))
  {
    mMatchFunc = func;
  }
  bool (MatchGlobalFwd::*mCutFunc)(const TrackLocMCH& mchTrack, const TrackLocMFT& mftTrack);
  void setCutFunction(bool (MatchGlobalFwd::*func)(const TrackLocMCH&, const TrackLocMFT&))
  {
    mCutFunc = func;
  }
  /// Matching methods
  /// Position
  double matchMFT_MCH_TracksXY(const TrackLocMCH& mchTrack, const TrackLocMFT& mftTrack);
  /// Position & Angles
  double matchMFT_MCH_TracksXYPhiTanl(const TrackLocMCH& mchTrack, const TrackLocMFT& mftTrack);
  /// Position, Angles & Charged Momentum
  double matchMFT_MCH_TracksAllParam(const TrackLocMCH& mchTrack, const TrackLocMFT& mftTrack);
  /// Hiroshima's Matching
  double matchHiroshima(const TrackLocMCH& mchTrack, const TrackLocMFT& mftTrack);

  /// Cut functions
  bool cutDisabled(const TrackLocMCH& mchTrack, const TrackLocMFT& mftTrack) { return true; };

  /// Converts mchTrack parameters to Forward coordinate system
  o2::dataformats::GlobalFwdTrack MCHtoFwd(const o2::mch::TrackParam& mchTrack);

  float mBz = -5.f;              ///< nominal Bz in kGauss
  float mMatchingPlaneZ = -77.5; ///< MCH-MFT matching plane Z position TODO: Make configurable
  Float_t mMFTDiskThicknessInX0 = 0.042 / 5;
  o2::InteractionRecord mStartIR{0, 0}; ///< IR corresponding to the start of the TF
  int mMFTROFrameLengthInBC = 0;        ///< MFT RO frame in BC (for MFT cont. mode only)
  float mMFTROFrameLengthMUS = -1.;     ///< MFT RO frame in \mus
  float mMFTROFrameLengthMUSInv = -1.;  ///< MFT RO frame in \mus inverse

  o2::BunchFilling mBunchFilling;
  std::array<int16_t, o2::constants::lhc::LHCMaxBunches> mClosestBunchAbove; // closest filled bunch from above
  std::array<int16_t, o2::constants::lhc::LHCMaxBunches> mClosestBunchBelow; // closest filled bunch from below

  bool mMFTTriggered = false; ///< MFT readout is triggered

  /// mapping for tracks' continuos ROF cycle to actual continuous readout ROFs with eventual gaps
  std::vector<int> mMFTTrackROFContMapping;

  const o2::globaltracking::RecoContainer* mRecoCont = nullptr;
  gsl::span<const o2::mch::TrackMCH> mMCHTracks;            ///< input MCH tracks
  gsl::span<const o2::mch::ROFRecord> mMCHTrackROFRec;      ///< MCH tracks ROFRecords
  gsl::span<const o2::mft::TrackMFT> mMFTTracks;            ///< input MFT tracks
  gsl::span<const o2::itsmft::ROFRecord> mMFTTrackROFRec;   ///< MFT tracks ROFRecords
  gsl::span<const int> mMFTTrackClusIdx;                    ///< input MFT track cluster indices span
  gsl::span<const o2::itsmft::ROFRecord> mMFTClusterROFRec; ///< input MFT clusters ROFRecord span
  gsl::span<const o2::MCCompLabel> mMFTTrkLabels;           ///< input MFT Track MC labels
  gsl::span<const o2::MCCompLabel> mMCHTrkLabels;           ///< input MCH Track MC labels

  std::vector<BracketF> mMCHROFTimes;                          ///< min/max times of MCH ROFs in \mus
  std::vector<TrackLocMCH> mMCHWork;                           ///< MCH track params prepared for matching
  std::vector<BracketF> mMFTROFTimes;                          ///< min/max times of MFT ROFs in \mus
  std::vector<TrackLocMFT> mMFTWork;                           ///< MFT track params prepared for matching
  std::vector<MFTCluster> mMFTClusters;                        ///< input MFT clusters
  std::vector<o2::dataformats::GlobalFwdTrack> mMatchedTracks; ///< MCH-MFT(-MID) Matched tracks
  std::vector<o2::MCCompLabel> mMatchLabels;                   ///< Output labels

  const o2::itsmft::TopologyDictionary* mMFTDict{nullptr}; // cluster patterns dictionary
  o2::itsmft::ChipMappingMFT mMFTMapping;
  bool mMCTruthON = false; ///< flag availability of MC truth
  TGeoManager* mGeoManager;
};

} // namespace globaltracking
} // namespace o2

#endif
