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

#include "HMPIDBase/Geo.h"
#include "DataFormatsHMP/Cluster.h"

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
/* enum from TOF::cluster
enum { kUpLeft = 0,    // 2^0, 1st bit
       kUp = 1,        // 2^1, 2nd bit
       kUpRight = 2,   // 2^2, 3rd bit
       kRight = 3,     // 2^3, 4th bit
       kDownRight = 4, // 2^4, 5th bit
       kDown = 5,      // 2^5, 6th bit
       kDownLeft = 6,  // 2^6, 7th bit
       kLeft = 7 };    // 2^7, 8th bit  */

/* ef: I can not get it to compile w/o commenting out this struct
///< original track in the currently loaded TPC-ITS reco output
struct TrackLocTPCITS : public o2::track::TrackParCov {
  o2::dataformats::EvIndex<int, int> source; ///< track origin id
  o2::math_utils::Bracket<float> timeBins;   ///< bracketing time-bins
  float zMin = 0;                            // min possible Z of this track
  float zMax = 0;                            // max possible Z of this track
  int matchID = MinusOne;                    ///< entry (none if MinusOne) of TOF matchTOF struct in the mMatchesTOF
  TrackLocTPCITS(const o2::track::TrackParCov& src, int tch, int tid) : o2::track::TrackParCov(src), source(tch, tid) {}
  TrackLocTPCITS() = default;
  ClassDefNV(TrackLocTPCITS, 1); // RS TODO: is this class needed?
};// */

class MatchHMP
{

  using Geo = o2::hmpid::Geo;
  using Cluster = o2::hmpid::Cluster; // ef: not hmp
  using evGIdx = o2::dataformats::EvIndex<int, o2::dataformats::GlobalTrackID>;
  using evIdx = o2::dataformats::EvIndex<int, int>;
  using timeEst = o2::dataformats::TimeStampWithError<float, float>;
  using matchTrack = std::pair<o2::track::TrackParCov, timeEst>;
  using trkType = o2::dataformats::MatchInfoTOFReco::TrackType;

 public:
  //  MatchHMP() = default;
  // ~MatchHMP() = default;
  ///< perform matching for provided input
  void run(const o2::globaltracking::RecoContainer& inp);

  ///< print settings
  void print() const;
  void printCandidatesHMP() const;

  ///< set time tolerance on track-HMP times comparison
  void setTimeTolerance(float val) { mTimeTolerance = val; }
  ///< get tolerance on track-HMP times comparison
  float getTimeTolerance() const { return mTimeTolerance; }

  ///< set space tolerance on track-HMP times comparison
  void setSpaceTolerance(float val) { mSpaceTolerance = val; }
  ///< get tolerance on track-HMP times comparison
  float getSpaceTolerance() const { return mSpaceTolerance; }

  ///< set number of sigma used to do the matching
  void setSigmaTimeCut(float val) { mSigmaTimeCut = val; }
  ///< get number of sigma used to do the matching
  float getSigmaTimeCut() const { return mSigmaTimeCut; }

  enum DebugFlagTypes : UInt_t {
    MatchTreeAll = 0x1 << 1, //  ///< produce matching candidates tree for all candidates
  };
  ///< check if partucular flags are set
  bool isDebugFlag(UInt_t flags) const { return mDBGFlags & flags; }

  ///< get debug trees flags
  UInt_t getDebugFlags() const { return mDBGFlags; }

  ///< set or unset debug stream flag
  void setDebugFlag(UInt_t flag, bool on = true);

  std::vector<o2::dataformats::MatchInfoHMP>& getMatchedTrackVector(trkType index)
  {
    return mMatchedTracks[index];
  }

  std::vector<o2::MCCompLabel>& getMatchedHMPLabelsVector(trkType index) { return mOutHMPLabels[index]; } ///< get vector of HMP label of matched tracks

  ///< set input TPC tracks cluster indices
  void setTPCTrackClusIdxInp(const gsl::span<const o2::tpc::TPCClRefElem> inp)
  {
    mTPCTrackClusIdx = inp;
  }

  ///< set input TPC cluster sharing map
  void setTPCClustersSharingMap(const gsl::span<const unsigned char> inp)
  {
    mTPCRefitterShMap = inp;
  }

  ///< set input TPC clusters
  void setTPCClustersInp(const o2::tpc::ClusterNativeAccess* inp)
  {
    mTPCClusterIdxStruct = inp;
  }

  void setFIT(bool value = true) { mIsFIT = value; }
  int findFITIndex(int bc);

  // bool makeConstrainedTPCTrack(int matchedID, o2::dataformats::TrackTPCHMP& trConstr); has not been declared
  bool makeConstrainedTPCTrack(int matchedID, o2::dataformats::TrackTPCTOF& trConstr);

  ///< populate externally provided container by TOF-time-constrained TPC tracks
  template <typename V>

  /* void makeConstrainedTPCTracks(V& container)
   {
     //checkRefitter();
     int nmatched = mMatchedTracks[trkType::TPC].size(), nconstrained = 0;
     container.resize(nmatched);
     for (unsigned i = 0; i < nmatched; i++) {
       if (makeConstrainedTPCTrack(i, container[nconstrained])) {
         nconstrained++;
       }
     }
     container.resize(nconstrained);
   }
  */

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
  void doMatching(int sec);

  bool propagateToRefX(o2::track::TrackParCov& trc, float xRef /*in cm*/, float stepInCm /*in cm*/, o2::track::TrackLTIntegral& intLT);
  bool propagateToRefXWithoutCov(o2::track::TrackParCov& trc, float xRef /*in cm*/, float stepInCm /*in cm*/, float bz);

  void updateTimeDependentParams();

  /*
  void addTPCSeed(const o2::tpc::TrackTPC& _tr, o2::dataformats::GlobalTrackID srcGID);
  void addITSTPCSeed(const o2::dataformats::TrackTPCITS& _tr, o2::dataformats::GlobalTrackID srcGID);
  void addTRDSeed(const o2::trd::TrackTRD& _tr, o2::dataformats::GlobalTrackID srcGID, float time0, float terr);
  void addConstrainedSeed(o2::track::TrackParCov& trc, o2::dataformats::GlobalTrackID srcGID, o2::track::TrackLTIntegral intLT0, timeEst timeMUS);
  void addTPCTRDSeed(const o2::track::TrackParCov& _tr, o2::dataformats::GlobalTrackID srcGID, int tpcID);
  void addITSTPCTRDSeed(const o2::track::TrackParCov& _tr, o2::dataformats::GlobalTrackID srcGID, int tpcID);
  */

  //================================================================

  // Data members
  const o2::globaltracking::RecoContainer* mRecoCont = nullptr;

  o2::InteractionRecord mStartIR{0, 0}; ///< IR corresponding to the start of the TF

  // for derived class
  int mCurrTracksTreeEntry = 0; ///< current tracks tree entry loaded to memory

  float mXRef = 500.; ///< reference radius to propage tracks for matching

  bool mMCTruthON = false; ///< flag availability of MC truth

  ///========== Parameters to be set externally, e.g. from CCDB ====================
  float mBz = 0;          ///< nominal Bz
  float mMaxInvPt = 999.; ///< derived from nominal Bz

  // to be done later
  float mTPCTBinMUS = 0.;    ///< TPC time bin duration in microseconds
  float mTPCTBinMUSInv = 0.; ///< inverse TPC time bin duration in microseconds
  float mTPCBin2Z = 0.;      ///< conversion coeff from TPC time-bin to Z

  bool mIsCosmics = false;    ///< switch on to reconstruct cosmics and match with TPC
  float mTimeTolerance = 1e3; ///< tolerance in ns for track-TOF time bracket matching
  float mSpaceTolerance = 10; ///< tolerance in cm for track-TOF time bracket matching
  int mSigmaTimeCut = 30.;    ///< number of sigmas to cut on time when matching the track to the TOF cluster

  bool mIsFIT = false;
  bool mIsITSTPCused = false;
  bool mIsTPCused = false;
  bool mIsTPCTRDused = false;
  bool mIsITSTPCTRDused = false;
  bool mSetHighPurity = false;

  unsigned long mTimestamp = 0; ///< in ms

  // from ruben
  gsl::span<const o2::tpc::TrackTPC> mTPCTracksArray; ///< input TPC tracks span

  ///>>>------ these are input arrays which should not be modified by the matching code
  //           since this info is provided by external device
  std::vector<o2::dataformats::TrackTPCITS> mITSTPCTracksArrayInp; ///< input tracks
  gsl::span<const Cluster> mHMPClustersArrayInp;                   ///< input HMPID clusters

  /// data needed for refit of time-constrained TPC tracks
  gsl::span<const o2::tpc::TPCClRefElem> mTPCTrackClusIdx;            ///< input TPC track cluster indices span
  gsl::span<const unsigned char> mTPCRefitterShMap;                   ///< externally set TPC clusters sharing map
  const o2::tpc::ClusterNativeAccess* mTPCClusterIdxStruct = nullptr; ///< struct holding the TPC cluster indices
  std::unique_ptr<o2::gpu::TPCFastTransform> mTPCTransform;           ///< TPC cluster transformation
  std::unique_ptr<o2::gpu::GPUO2InterfaceRefit> mTPCRefitter;         ///< TPC refitter used for TPC tracks refit during the reconstruction

  const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mHMPClusLabels; ///< input HMP clusters MC labels (pointer to read from tree)

  int mNotPropagatedToHMP[trkType::SIZE]; ///< number of tracks failing in propagation

  ///< working copy of the input tracks
  std::vector<matchTrack> mTracksWork[trkType::SIZE];                   ///< track params prepared for matching + time value
  std::vector<o2::MCCompLabel> mTracksLblWork[trkType::SIZE];           ///< TPCITS track labels
  std::vector<o2::track::TrackLTIntegral> mLTinfos[trkType::SIZE];      ///< expected times and others
  std::vector<o2::dataformats::GlobalTrackID> mTrackGid[trkType::SIZE]; ///< expected times and others
  ///< per sector indices of track entry in mTracksWork
  std::array<std::vector<int>, o2::constants::math::NSectors> mTracksSectIndexCache[trkType::SIZE];

  std::vector<float> mExtraTPCFwdTime; ///< track extra params for TPC tracks: Fws Max time
  std::vector<Cluster> mHMPClusWork;   ///< track params prepared for matching

  std::vector<int8_t> mSideTPC; ///< track side for TPC tracks

  ///< per sector indices of HMP cluster entry in mHMPClusWork
  std::array<std::vector<int>, o2::constants::math::NSectors> mHMPClusSectIndexCache;

  ///< array of track-HMPCluster pairs from the matching
  std::vector<o2::dataformats::MatchInfoHMP> mMatchedTracksPairs;

  ///< array of matched HMPCluster with matching information (residuals, expected times...) with the corresponding vector of indices
  std::vector<o2::dataformats::MatchInfoHMP> mMatchedTracks[trkType::SIZEALL]; // this is the output of the matching -> UNCONS, CONSTR
  std::vector<o2::MCCompLabel> mOutHMPLabels[trkType::SIZEALL];                ///< TOF label of matched tracks

  std::vector<int> mMatchedTracksIndex[trkType::SIZE]; // vector of indexes of the tracks to be matched

  // ef : move to cxx
  // int mNumOfClusters;                   // number of clusters to be matched
  // std::unique_ptr<int[]> mMatchedClustersIndex; //[mNumOfClusters] ef : change to smart-pointer
  // int* mMatchedClustersIndex = nullptr; //[mNumOfClusters]

  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  UInt_t mDBGFlags = 0;

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
