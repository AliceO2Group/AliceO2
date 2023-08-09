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

#ifndef ALICEO2_GLOBTRACKING_MATCHHMP_
#define ALICEO2_GLOBTRACKING_MATCHHMP_

#include <Rtypes.h>
#include <array>
#include <vector>
#include <string>
#include <gsl/span>
#include <TStopwatch.h>
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/TrackTPCTOF.h"
#include "ReconstructionDataFormats/MatchInfoTOFReco.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "CommonDataFormat/EvIndex.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "CommonUtils/TreeStreamRedirector.h"

#include "SimulationDataFormat/MCTruthContainer.h"
#include "DetectorsBase/Propagator.h"
#include "MathUtils/Cartesian.h"
#include "MathUtils/Utils.h"
#include "CommonConstants/MathConstants.h"
#include "CommonConstants/PhysicsConstants.h"
#include "CommonConstants/GeomConstants.h"
#include "DetectorsBase/GeometryManager.h"

#include "DataFormatsHMP/Cluster.h"
#include "GlobalTracking/MatchTPCITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTRD/TrackTRD.h"
#include "ReconstructionDataFormats/PID.h"
#include "TPCFastTransform.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "ReconstructionDataFormats/MatchInfoHMP.h"
#include "ReconstructionDataFormats/TrackHMP.h"

#include "HMPIDBase/Geo.h"
#include "DataFormatsHMP/Cluster.h"
#include "DataFormatsHMP/Trigger.h"

namespace o2
{

namespace globaltracking
{
class RecoContainer;
}

namespace dataformats
{
template <typename TruthElement>
class MCTruthContainer;
}

namespace globaltracking
{

class MatchHMP
{

  using Geo = o2::hmpid::Geo;
  using Cluster = o2::hmpid::Cluster;
  using Trigger = o2::hmpid::Trigger;
  using evGIdx = o2::dataformats::EvIndex<int, o2::dataformats::GlobalTrackID>;
  using evIdx = o2::dataformats::EvIndex<int, int>;
  using timeEst = o2::dataformats::TimeStampWithError<float, float>;
  using matchTrack = std::pair<o2::track::TrackParCov, timeEst>;

 public:
  ///< perform matching for provided input
  void run(const o2::globaltracking::RecoContainer& inp);

  ///< print settings
  void print() const;
  void printCandidatesHMP() const;

  enum DebugFlagTypes : UInt_t {
    MatchTreeAll = 0x1 << 1, //  ///< produce matching candidates tree for all candidates
  };

  enum trackType : int8_t { UNCONS = 0,
                            CONSTR,
                            SIZE,
                            TPC = 0,
                            ITSTPC,
                            TPCTRD,
                            ITSTPCTRD,
                            SIZEALL };

  std::vector<o2::dataformats::MatchInfoHMP>& getMatchedTrackVector(o2::globaltracking::MatchHMP::trackType index) { return mMatchedTracks[index]; }

  std::vector<o2::MCCompLabel>& getMatchedHMPLabelsVector(o2::globaltracking::MatchHMP::trackType index) { return mOutHMPLabels[index]; } ///< get vector of HMP label of matched tracks

  void setTS(unsigned long creationTime)
  {
    mTimestamp = creationTime;
  }
  unsigned long getTS() const { return mTimestamp; }

 private:
  // bool prepareFITData();
  int prepareInteractionTimes();
  bool prepareTracks();
  bool prepareHMPClusters();
  void doFastMatching();
  void doMatching();

  static int intTrkCha(o2::track::TrackParCov* pTrk, double& xPc, double& yPc, double& xRa, double& yRa, double& theta, double& phi, double bz);               // find track-PC intersection, retuns chamber ID
  static int intTrkCha(int ch, o2::dataformats::TrackHMP* pHmpTrk, double& xPc, double& yPc, double& xRa, double& yRa, double& theta, double& phi, double bz); // find track-PC intersection, retuns chamber ID

  bool intersect(Double_t pnt[3], Double_t norm[3]) const;

  void addTPCSeed(const o2::tpc::TrackTPC& _tr, o2::dataformats::GlobalTrackID srcGID, float time0, float terr);
  void addITSTPCSeed(const o2::dataformats::TrackTPCITS& _tr, o2::dataformats::GlobalTrackID srcGID, float time0, float terr);
  void addTRDSeed(const o2::trd::TrackTRD& _tr, o2::dataformats::GlobalTrackID srcGID, float time0, float terr);
  void addTPCTOFSeed(const o2::dataformats::TrackTPCTOF& _tr, o2::dataformats::GlobalTrackID srcGID, float time0, float terr);
  void addConstrainedSeed(o2::track::TrackParCov& trc, o2::dataformats::GlobalTrackID srcGID, timeEst timeMUS);

  // Data members
  const o2::globaltracking::RecoContainer* mRecoCont = nullptr;

  o2::InteractionRecord mStartIR{0, 0}; ///< IR corresponding to the start of the TF

  // for derived class
  int mCurrTracksTreeEntry = 0; ///< current tracks tree entry loaded to memory

  bool mMCTruthON = false; ///< flag availability of MC truth

  ///========== Parameters to be set externally, e.g. from CCDB ====================
  float mBz = 0;          ///< nominal Bz
  float mMaxInvPt = 999.; ///< derived from nominal Bz

  // to be done later
  float mTPCTBinMUS = 0.;               ///< TPC time bin duration in microseconds
  float mTPCTBinMUSInv = 0.;            ///< inverse TPC time bin duration in microseconds
  float mTPCBin2Z = 0.;                 ///< conversion coeff from TPC time-bin to Z
  float mTimeTolerance = 1e3;           ///< tolerance in ns for track-TOF time bracket matching
  float mExtraTimeToleranceTRD = 500E3; ///< extra tolerance in ns for track-TOF time bracket matching
  float mExtraTimeToleranceTOF = 500E3; ///< extra tolerance in ns for track-TOF time bracket matching
  float mSigmaTimeCut = 1.;             ///< number of sigmas to cut on time when matching the track to the TOF cluster

  static constexpr Double_t BC_TIME = o2::constants::lhc::LHCBunchSpacingNS; // bunch crossing in ns
  static constexpr Double_t BC_TIME_INV = 1. / BC_TIME;                      // inv bunch crossing in ns
  static constexpr Double_t BC_TIME_INPS = BC_TIME * 1000;                   // bunch crossing in ps
  static constexpr Double_t BC_TIME_INPS_INV = 1. / BC_TIME_INPS;            // inv bunch crossing in ps

  bool mIsFIT = false;
  bool mIsTPCused = false;
  bool mIsITSTPCused = false;
  bool mIsTPCTOFused = false;
  bool mIsTPCTRDused = false;
  bool mIsITSTPCTOFused = false;
  bool mIsTPCTRDTOFused = false;
  bool mIsITSTPCTRDused = false;
  bool mIsITSTPCTRDTOFused = false;
  bool mSetHighPurity = false;

  float mTPCVDrift = -1.; ///< TPC drift speed in cm/microseconds

  unsigned long mTimestamp = 0; ///< in ms

  ///>>>------ these are input arrays which should not be modified by the matching code
  //           since this info is provided by external device
  std::vector<o2::dataformats::TrackTPCITS> mITSTPCTracksArrayInp; ///< input tracks
  gsl::span<const Cluster> mHMPClustersArray;                      ///< input HMPID clusters
  gsl::span<const Trigger> mHMPTriggersArray;                      ///< input HMPID triggers

  const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mHMPClusLabels; ///< input HMP clusters MC labels (pointer to read from tree)

  int mNotPropagatedToHMP[o2::globaltracking::MatchHMP::trackType::SIZE]; ///< number of tracks failing in propagation

  ///< working copy of the input tracks
  std::vector<matchTrack> mTracksWork[o2::globaltracking::MatchHMP::trackType::SIZE]; ///< track params prepared for matching + time value
  std::vector<Trigger> mHMPTriggersWork;
  std::vector<o2::MCCompLabel> mTracksLblWork[o2::globaltracking::MatchHMP::trackType::SIZE]; ///< track labels

  std::vector<int> mTracksIndexCache[o2::globaltracking::MatchHMP::trackType::SIZE]; ///< indices of track entry in mTracksWork
  std::vector<int> mHMPTriggersIndexCache;                                           ///< indices of track entry in mHMPTriggersWork

  ///< array of matched HMPCluster with matching information
  std::vector<o2::dataformats::MatchInfoHMP> mMatchedTracks[o2::globaltracking::MatchHMP::trackType::SIZE]; // this is the output of the matching -> UNCONS, CONSTR
  std::vector<o2::MCCompLabel> mOutHMPLabels[o2::globaltracking::MatchHMP::trackType::SIZE];                ///< HMP label of matched tracks

  std::vector<o2::dataformats::GlobalTrackID> mTrackGid[o2::globaltracking::MatchHMP::trackType::SIZE]; ///< expected times and others
  std::vector<int> mMatchedTracksIndex[o2::globaltracking::MatchHMP::trackType::SIZE];                  // vector of indexes of the tracks to be matched

  int mNumOfTriggers; // number of HMP triggers

  ///----------- aux stuff --------------///
  static constexpr float MAXSNP = 0.85; // max snp of ITS or TPC track at xRef to be matched

  TStopwatch mTimerTot;
  TStopwatch mTimerMatchITSTPC;
  TStopwatch mTimerMatchTPC;
  TStopwatch mTimerDBG;

  ClassDef(MatchHMP, 1);
};
} // namespace globaltracking
} // namespace o2

#endif
