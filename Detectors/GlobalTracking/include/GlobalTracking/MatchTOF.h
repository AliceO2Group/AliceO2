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

/// \file MatchTOF.h
/// \brief Class to perform TOF matching to global tracks
/// \author Francesco.Noferini@cern.ch, Chiara.Zampolli@cern.ch

#ifndef ALICEO2_GLOBTRACKING_MATCHTOF_
#define ALICEO2_GLOBTRACKING_MATCHTOF_

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
#include "DataFormatsTOF/CalibInfoTOF.h"
#include "CommonDataFormat/EvIndex.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "TOFBase/Geo.h"
#include "DataFormatsTOF/Cluster.h"
#include "GlobalTracking/MatchTPCITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "ReconstructionDataFormats/PID.h"
#include "TPCFastTransform.h"

// from FIT
#include "DataFormatsFT0/RecPoints.h"

namespace o2
{

namespace dataformats
{
template <typename TruthElement>
class MCTruthContainer;
}

namespace globaltracking
{

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
};

class MatchTOF
{
  using Geo = o2::tof::Geo;
  using Cluster = o2::tof::Cluster;
  using evGIdx = o2::dataformats::EvIndex<int, o2::dataformats::GlobalTrackID>;
  using evIdx = o2::dataformats::EvIndex<int, int>;
  using timeEst = o2::dataformats::TimeStampWithError<float, float>;
  using matchTrack = std::pair<o2::track::TrackParCov, timeEst>;

 public:
  ///< perform matching for provided input
  virtual void run();

  void setCosmics()
  {
    mIsCosmics = true;
    mSpaceTolerance = 150;
    mTimeTolerance = 50e3;
  }

  void setHighPurity(bool value = true) { mSetHighPurity = value; }

  ///< attach DPL data and run
  void setTOFClusterArray(const gsl::span<const Cluster>& clusterArray, const o2::dataformats::MCTruthContainer<o2::MCCompLabel>& toflab);
  void setTPCTrackArray(const gsl::span<const o2::tpc::TrackTPC>& trackArray, const gsl::span<const o2::MCCompLabel>& tpclab);
  void setITSTPCTrackArray(const gsl::span<const o2::dataformats::TrackTPCITS>& trackArray, const gsl::span<const o2::MCCompLabel>& tpclab);

  ///< print settings
  void print() const;
  void printCandidatesTOF() const;

  ///< set time tolerance on track-TOF times comparison
  void setTimeTolerance(float val) { mTimeTolerance = val; }
  ///< get tolerance on track-TOF times comparison
  float getTimeTolerance() const { return mTimeTolerance; }

  ///< set space tolerance on track-TOF times comparison // this in the old AliRoot was the TOF matching window
  void setSpaceTolerance(float val) { mSpaceTolerance = val; }
  ///< get tolerance on track-TOF times comparison
  float getSpaceTolerance() const { return mSpaceTolerance; }

  ///< set number of sigma used to do the matching
  void setSigmaTimeCut(float val) { mSigmaTimeCut = val; }
  ///< get number of sigma used to do the matching
  float getSigmaTimeCut() const { return mSigmaTimeCut; }

  enum DebugFlagTypes : UInt_t {
    MatchTreeAll = 0x1 << 1, ///< produce matching candidates tree for all candidates
  };
  ///< check if partucular flags are set
  bool isDebugFlag(UInt_t flags) const { return mDBGFlags & flags; }

  ///< get debug trees flags
  UInt_t getDebugFlags() const { return mDBGFlags; }

  ///< set or unset debug stream flag
  void setDebugFlag(UInt_t flag, bool on = true);

  ///< set the name of output debug file
  void setDebugTreeFileName(std::string name)
  {
    if (!name.empty()) {
      mDebugTreeFileName = name;
    }
  }

  std::vector<o2::dataformats::MatchInfoTOF>& getMatchedTrackVector(o2::dataformats::MatchInfoTOFReco::TrackType index) { return mMatchedTracks[index]; }
  std::vector<o2::dataformats::CalibInfoTOF>& getCalibVector() { return mCalibInfoTOF; }

  std::vector<o2::MCCompLabel>& getMatchedTOFLabelsVector(o2::dataformats::MatchInfoTOFReco::TrackType index) { return mOutTOFLabels[index]; } ///< get vector of TOF label of matched tracks

  // this method is deprecated
  void setFITRecPoints(const std::vector<o2::ft0::RecPoints>* recpoints)
  {
    if (recpoints) {
      mFITRecPoints = {recpoints->data(), recpoints->size()};
    }
  }
  void setFITRecPoints(gsl::span<o2::ft0::RecPoints const> recpoints)
  {
    mFITRecPoints = recpoints;
  }

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

  int findFITIndex(int bc);

  ///< populate externally provided container by TOF-time-constrained TPC tracks
  template <typename V>
  void makeConstrainedTPCTracks(V& container)
  {
    checkRefitter();
    int nmatched = mMatchedTracks[o2::dataformats::MatchInfoTOFReco::TrackType::TPC].size(), nconstrained = 0;
    container.resize(nmatched);
    for (unsigned i = 0; i < nmatched; i++) {
      if (makeConstrainedTPCTrack(i, container[nconstrained])) {
        nconstrained++;
      }
    }
    container.resize(nconstrained);
  }

 protected:
  virtual bool prepareTracks();
  virtual bool prepareTPCTracks();
  virtual bool prepareTOFClusters();

  void doMatching(int sec, o2::dataformats::MatchInfoTOFReco::TrackType type = o2::dataformats::MatchInfoTOFReco::TrackType::ITSTPC);
  void doMatchingForTPC(int sec);
  void selectBestMatches();
  void selectBestMatchesHP();
  bool propagateToRefX(o2::track::TrackParCov& trc, float xRef /*in cm*/, float stepInCm /*in cm*/, o2::track::TrackLTIntegral& intLT);
  bool propagateToRefXWithoutCov(o2::track::TrackParCov& trc, float xRef /*in cm*/, float stepInCm /*in cm*/, float bz);

  void updateTimeDependentParams();
  void checkRefitter();
  bool makeConstrainedTPCTrack(int matchedID, o2::dataformats::TrackTPCTOF& trConstr);

  //================================================================

  // Data members

  // for derived class
  int mCurrTracksTreeEntry = 0;  ///< current tracks tree entry loaded to memory
  bool mSAInitDone = false;      ///< flag that standalone init already done
  bool mWFInputAttached = false; ///< flag that the standalone input is attached
  Bool_t mIsworkflowON = kFALSE; ///< flag used in the standalone mode

  float mXRef = Geo::RMIN; ///< reference radius to propage tracks for matching

  bool mMCTruthON = false; ///< flag availability of MC truth

  ///========== Parameters to be set externally, e.g. from CCDB ====================
  float mBz = 0; ///< nominal Bz

  // to be done later
  float mTPCTBinMUS = 0.;    ///< TPC time bin duration in microseconds
  float mTPCTBinMUSInv = 0.; ///< inverse TPC time bin duration in microseconds
  float mTPCBin2Z = 0.;      ///< conversion coeff from TPC time-bin to Z

  bool mIsCosmics = false;    ///< switch on to reconstruct cosmics and match with TPC
  float mTimeTolerance = 1e3; ///< tolerance in ns for track-TOF time bracket matching
  float mSpaceTolerance = 10; ///< tolerance in cm for track-TOF time bracket matching
  int mSigmaTimeCut = 30.;    ///< number of sigmas to cut on time when matching the track to the TOF cluster

  bool mIsITSTPCused = false;
  bool mIsTPCused = false;
  bool mIsTPCTRDused = false;
  bool mIsITSTPCTRDused = false;
  bool mSetHighPurity = false;

  ///>>>------ these are input arrays which should not be modified by the matching code
  //           since this info is provided by external device
  gsl::span<const o2::dataformats::TrackTPCITS> mITSTPCTracksArrayInp;  ///< input tracks
  std::vector<o2::dataformats::TrackTPCITS>* mITSTPCTracksArrayInpVect; ///< input tracks (vector to read from tree)
  gsl::span<const o2::tpc::TrackTPC> mTPCTracksArrayInp;                ///< input TPC tracks
  std::vector<o2::tpc::TrackTPC>* mTPCTracksArrayInpVect;               ///< input tracks (vector to read from tree)
  gsl::span<const Cluster> mTOFClustersArrayInp;                        ///< input TOF clusters
  std::vector<Cluster>* mTOFClustersArrayInpVect;                       ///< input TOF clusters (vector to read from tree)

  /// data needed for refit of time-constrained TPC tracks
  gsl::span<const o2::tpc::TPCClRefElem> mTPCTrackClusIdx;            ///< input TPC track cluster indices span
  gsl::span<const unsigned char> mTPCRefitterShMap;                   ///< externally set TPC clusters sharing map
  const o2::tpc::ClusterNativeAccess* mTPCClusterIdxStruct = nullptr; ///< struct holding the TPC cluster indices
  std::unique_ptr<o2::gpu::TPCFastTransform> mTPCTransform;           ///< TPC cluster transformation
  std::unique_ptr<o2::gpu::GPUO2InterfaceRefit> mTPCRefitter;         ///< TPC refitter used for TPC tracks refit during the reconstruction

  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mTOFClusLabels;     ///< input TOF clusters MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mTOFClusLabelsPtr; ///< input TOF clusters MC labels (pointer to read from tree)
  std::vector<o2::MCCompLabel> mTracksLblWork;                           ///<TPCITS track labels

  gsl::span<const o2::MCCompLabel> mTPCLabels[o2::dataformats::MatchInfoTOFReco::TrackType::SIZE];  ///< TPC label of input tracks
  std::vector<o2::MCCompLabel>* mTPCLabelsVect[o2::dataformats::MatchInfoTOFReco::TrackType::SIZE]; ///< TPC label of input tracks (vector to read from tree)

  gsl::span<o2::ft0::RecPoints const> mFITRecPoints; ///< FIT recpoints

  /// <<<-----

  ///<working copy of the input tracks
  std::vector<matchTrack> mTracksWork[o2::dataformats::MatchInfoTOFReco::TrackType::SIZE]; ///<track params prepared for matching + time value

  std::vector<float> mExtraTPCFwdTime;              ///<track extra params for TPC tracks: Fws Max time
  std::vector<o2::track::TrackLTIntegral> mLTinfos; ///<expected times and others
  std::vector<Cluster> mTOFClusWork;                ///<track params prepared for matching
  std::vector<int8_t> mSideTPC;                     ///<track side for TPC tracks

  ///< per sector indices of track entry in mTracksWork
  std::array<std::vector<int>, o2::constants::math::NSectors> mTracksSectIndexCache[o2::dataformats::MatchInfoTOFReco::TrackType::SIZE];
  ///< per sector indices of TOF cluster entry in mTOFClusWork
  std::array<std::vector<int>, o2::constants::math::NSectors> mTOFClusSectIndexCache;

  ///<array of track-TOFCluster pairs from the matching
  std::vector<o2::dataformats::MatchInfoTOFReco> mMatchedTracksPairs;

  ///<array of TOFChannel calibration info
  std::vector<o2::dataformats::CalibInfoTOF> mCalibInfoTOF;

  ///<array of matched TOFCluster with matching information (residuals, expected times...) with the corresponding vector of indices
  //std::vector<o2::dataformats::MatchInfoTOF> mMatchedTracks;
  std::vector<o2::dataformats::MatchInfoTOF> mMatchedTracks[o2::dataformats::MatchInfoTOFReco::TrackType::SIZE]; // this is the output of the matching
  std::vector<o2::MCCompLabel> mOutTOFLabels[o2::dataformats::MatchInfoTOFReco::TrackType::SIZE];                ///< TOF label of matched tracks

  int mNumOfTracks[o2::dataformats::MatchInfoTOFReco::TrackType::SIZE];                     // number of tracks to be matched
  std::vector<int> mMatchedTracksIndex[o2::dataformats::MatchInfoTOFReco::TrackType::SIZE]; // vector of indexes of the tracks to be matched

  int mNumOfClusters;                   // number of clusters to be matched
  int* mMatchedClustersIndex = nullptr; //[mNumOfClusters]

  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  UInt_t mDBGFlags = 0;
  std::string mDebugTreeFileName = "dbg_matchTOF.root"; ///< name for the debug tree file

  ///----------- aux stuff --------------///
  static constexpr float MAXSNP = 0.85; // max snp of ITS or TPC track at xRef to be matched

  TStopwatch mTimerTot;
  TStopwatch mTimerDBG;
  ClassDefNV(MatchTOF, 1);
};
} // namespace globaltracking
} // namespace o2

#endif
