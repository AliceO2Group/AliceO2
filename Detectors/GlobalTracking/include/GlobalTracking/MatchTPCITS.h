// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MatchTPCITS.h
/// \brief Class to perform TPC ITS matching
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_GLOBTRACKING_MATCHTPCITS_
#define ALICEO2_GLOBTRACKING_MATCHTPCITS_

#define _ALLOW_DEBUG_TREES_ // to allow debug and control tree output

#include <Rtypes.h>
#include <array>
#include <vector>
#include <string>
#include <TStopwatch.h>
#include "DataFormatsTPC/TrackTPC.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "CommonDataFormat/EvIndex.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsFT0/RecPoints.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "TPCFastTransform.h"
#include "GPUO2Interface.h" // Needed for propper settings in GPUParam.h
#include "GPUParam.h"       // Consider more universal access

class TTree;

namespace o2
{

namespace dataformats
{
template <typename TruthElement>
class MCTruthContainer;
}

namespace its
{
class TrackITS;
}

namespace itsmft
{
class Cluster;
}

namespace tpc
{
class TrackTPC;
}

namespace globaltracking
{

constexpr int MinusOne = -1;
constexpr int Validated = -2;

///< flags to tell the status of TPC-ITS tracks comparison
enum TrackRejFlag : int {
  Accept = 0,
  RejectOnY, // rejected comparing DY difference of tracks
  RejectOnZ,
  RejectOnSnp,
  RejectOnTgl,
  RejectOnQ2Pt,
  RejectOnChi2,
  NSigmaShift = 10
};

///< timing (in TPC time-bins) bracket assumed for the track
struct TimeBracket {
  float tmin = 0.f; ///< min possible time(bin)
  float tmax = 0.f; ///< max possible time(bin)
  TimeBracket() = default;
  TimeBracket(float mn, float mx) : tmin(mn), tmax(mx) {}
  void set(float tmn, float tmx)
  {
    tmin = tmn;
    tmax = tmx;
  }
  ClassDefNV(TimeBracket, 1);
};

///< TPC track parameters propagated to reference X, with time bracket and index of
///< original track in the currently loaded TPC reco output
struct TrackLocTPC : public o2::track::TrackParCov {
  o2::dataformats::EvIndex<int, int> source; ///< track origin id
  TimeBracket timeBins;                      ///< bracketing time-bins
  float zMin = 0;                            // min possible Z of this track
  float zMax = 0;                            // max possible Z of this track
  int matchID = MinusOne;                    ///< entry (non if MinusOne) of its matchTPC struct in the mMatchesTPC
  TrackLocTPC(const o2::track::TrackParCov& src, int tch, int tid) : o2::track::TrackParCov(src), source(tch, tid) {}
  TrackLocTPC() = default;
  ClassDefNV(TrackLocTPC, 1);
};

///< ITS track outward parameters propagated to reference X, with time bracket and index of
///< original track in the currently loaded ITS reco output
struct TrackLocITS : public o2::track::TrackParCov {
  o2::dataformats::EvIndex<int, int> source; ///< track origin id
  int roFrame = MinusOne;                    ///< ITS readout frame assigned to this track
  int matchID = MinusOne;                    ///< entry (non if MinusOne) of its matchCand struct in the mMatchesITS
  TrackLocITS(const o2::track::TrackParCov& src, int tch, int tid) : o2::track::TrackParCov(src), source(tch, tid) {}
  TrackLocITS() = default;
  ClassDefNV(TrackLocITS, 1);
};

///< each TPC or ITS track having at least 1 matching ITS or TPC candidate records
///< in the matchCandidate the ID of the 1st (best) matchRecord in the mMatchRecordsITS
///< ot mMatchRecordsTPC container
struct matchCand {
  o2::dataformats::EvIndex<int, int> source; ///< track origin id
  int first = MinusOne;                      ///< 1st match for this track in the mMatchRecordsTPC
  matchCand(const o2::dataformats::EvIndex<int, int>& src) : source(src) {}
  matchCand() = default;
};

///< record TPC or ITS track associated with single ITS or TPC track and reference on
///< the next (worse chi2) matchRecord of the same TPC or ITS track
struct matchRecord {
  float chi2 = -1.f;        ///< matching chi2
  int matchID = MinusOne;   ///< id of parnter matchCand struct in mMatchesITS/TPC container
  int nextRecID = MinusOne; ///< index of eventual next record

  matchRecord(int mtcID, float chi2match) : matchID(mtcID), chi2(chi2match) {}
  matchRecord(int mtcID, float chi2match, int nxt) : matchID(mtcID), chi2(chi2match), nextRecID(nxt) {}
  matchRecord() = default;
};

class MatchTPCITS
{
  using MCLabCont = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  using TPCTransform = o2::gpu::TPCFastTransform;

 public:
  static constexpr float XTPCInnerRef = 83.0;                            ///< reference radius at which TPC provides the tracks
  static constexpr float XTPCOuterRef = 255.0;                           ///< reference radius to propagate outer TPC track
  static constexpr float XMatchingRef = 70.0;                            ///< reference radius to propage tracks for matching
  static constexpr float YMaxAtXMatchingRef = XMatchingRef * 0.17632698; ///< max Y in the sector at reference X

  ///< perform matching for provided input
  void run();

  ///< perform all initializations
  void init();

  ///< clear results of previous event reco
  void clear();

  ///< set InteractionRecods for the beginning of the TF
  void setStartIR(const o2::InteractionRecord& ir) { mStartIR = ir; }

  ///< set input source: trees or DPL
  void setDPLIO(bool v);
  bool isDPLIO() const { return mDPLIO; }

  ///< ITS readout mode
  void setITSTriggered(bool v) { mITSTriggered = v; }
  bool isITSTriggered() const { return mITSTriggered; }

  ///< set ITS ROFrame duration in microseconds
  void setITSROFrameLengthMUS(float fums) { mITSROFrameLengthMUS = fums; }

  ///< set ITS 0-th ROFrame time start in \mus
  void setITSROFrameOffsetMUS(float v) { mITSROFrameOffsetMUS = v; }

  // ==================== >> DPL-driven input >> =======================

  ///< set flag to use MC truth from the DPL input
  void setMCTruthOn(bool v)
  {
    assertDPLIO(true);
    mMCTruthON = v;
  }

  ///< set input ITS tracks received via DPL
  void setITSTracksInp(const std::vector<o2::its::TrackITS>* inp)
  {
    assertDPLIO(true);
    mITSTracksArrayInp = inp;
  }

  ///< set input ITS tracks cluster indices received via DPL
  void setITSTrackClusIdxInp(const std::vector<int>* inp)
  {
    assertDPLIO(true);
    mITSTrackClusIdxInp = inp;
  }

  ///< set input ITS tracks cluster indices received via DPL
  void setITSTrackClusIdxInp(gsl::span<const int> inp)
  {
    assertDPLIO(true);
    mITSTrackClusIdxSPAN = inp;
  }

  ///< set input ITS tracks ROF records received via DPL
  void setITSTrackROFRecInp(const std::vector<o2::itsmft::ROFRecord>* inp)
  {
    assertDPLIO(true);
    mITSTrackROFRec = inp;
  }

  ///< set input ITS clusters received via DPL
  void setITSClustersInp(const std::vector<o2::itsmft::Cluster>* inp)
  {
    assertDPLIO(true);
    mITSClustersArrayInp = inp;
  }

  ///< set input ITS clusters ROF records received via DPL
  void setITSClusterROFRecInp(const std::vector<o2::itsmft::ROFRecord>* inp)
  {
    assertDPLIO(true);
    mITSClusterROFRec = inp;
  }

  ///< set input TPC tracks received via DPL
  void setTPCTracksInp(const std::vector<o2::tpc::TrackTPC>* inp)
  {
    assertDPLIO(true);
    mTPCTracksArrayInp = inp;
  }

  ///< set input TPC clusters received via DPL
  void setTPCClustersInp(const o2::tpc::ClusterNativeAccess* inp)
  {
    assertDPLIO(true);
    mTPCClusterIdxStruct = inp;
  }

  ///< set input ITS track MC labels received via DPL
  void setITSTrkLabelsInp(const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* lbl)
  {
    assertDPLIO(true);
    mITSTrkLabels = lbl;
  }

  ///< set input TPC track MC labels received via DPL
  void setTPCTrkLabelsInp(const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* lbl)
  {
    assertDPLIO(true);
    mTPCTrkLabels = lbl;
  }

  ///< set input FIT info received via DPL
  void setFITInfoInp(const std::vector<o2::ft0::RecPoints>* inp)
  {
    assertDPLIO(true);
    mFITInfoInp = inp;
  }

  // ===================== << DPL-driven input << ========================

  ///< set tree/chain containing ITS tracks
  void setInputTreeITSTracks(TTree* tree) { mTreeITSTracks = tree; }

  ///< set tree/chain containing ITS ROFRecs
  void setInputTreeITSTrackROFRec(TTree* tree) { mTreeITSTrackROFRec = tree; }

  ///< set tree/chain containing TPC tracks
  void setInputTreeTPCTracks(TTree* tree) { mTreeTPCTracks = tree; }

  ///< set tree/chain containing ITS clusters
  void setInputTreeITSClusters(TTree* tree) { mTreeITSClusters = tree; }

  ///< set tree/chain containing ITS cluster ROF records
  void setInputTreeITSClusterROFRec(TTree* tree) { mTreeITSClusterROFRec = tree; }

  ///< set optional input for FIT info
  void setInputTreeFITInfo(TTree* tree) { mTreeFITInfo = tree; }

  ///< set reader for TPC clusters
  void setInputTPCClustersReader(o2::tpc::ClusterNativeHelper::Reader* reader) { mTPCClusterReader = reader; }

  ///< set output tree to write matched tracks
  void setOutputTree(TTree* tr) { mOutputTree = tr; }

  ///< set input branch names for the input from the tree
  void setITSTrackBranchName(const std::string& nm) { mITSTrackBranchName = nm; }
  void setTPCTrackBranchName(const std::string& nm) { mTPCTrackBranchName = nm; }
  void setITSClusterBranchName(const std::string& nm) { mITSClusterBranchName = nm; }
  void setITSMCTruthBranchName(const std::string& nm) { mITSMCTruthBranchName = nm; }
  void setTPCMCTruthBranchName(const std::string& nm) { mTPCMCTruthBranchName = nm; }
  void setFITInfoBbranchName(const std::string& nm) { mFITInfoBranchName = nm; }
  void setOutTPCITSTracksBranchName(const std::string& nm) { mOutTPCITSTracksBranchName = nm; }
  void setOutTPCMCTruthBranchName(const std::string& nm) { mOutTPCMCTruthBranchName = nm; }
  void setOutITSMCTruthBranchName(const std::string& nm) { mOutITSMCTruthBranchName = nm; }

  ///< get input branch names for the input from the tree
  const std::string& getITSTrackBranchName() const { return mITSTrackBranchName; }
  const std::string& getTPCTrackBranchName() const { return mTPCTrackBranchName; }
  const std::string& getITSClusterBranchName() const { return mITSClusterBranchName; }
  const std::string& getITSMCTruthBranchName() const { return mITSMCTruthBranchName; }
  const std::string& getTPCMCTruthBranchName() const { return mTPCMCTruthBranchName; }
  const std::string& getFITInfoBranchName() const { return mFITInfoBranchName; }
  const std::string& getOutTPCITSTracksBranchName() const { return mOutTPCITSTracksBranchName; }
  const std::string& getOutTPCMCTruthBranchName() const { return mOutTPCMCTruthBranchName; }
  const std::string& getOutITSMCTruthBranchName() const { return mOutITSMCTruthBranchName; }

  ///< print settings
  void print() const;
  void printCandidatesTPC() const;
  void printCandidatesITS() const;

  std::vector<o2::dataformats::TrackTPCITS>& getMatchedTracks() { return mMatchedTracks; }
  std::vector<o2::MCCompLabel>& getMatchedITSLabels() { return mOutITSLabels; }
  std::vector<o2::MCCompLabel>& getMatchedTPCLabels() { return mOutTPCLabels; }

  //>>> ====================== cuts ================================>>>

  ///< set cuts on absolute difference of ITS vs TPC track parameters
  void setCrudeAbsDiffCut(const std::array<float, o2::track::kNParams>& vals) { mCrudeAbsDiffCut = vals; }
  ///< get cuts on absolute difference of ITS vs TPC track parameters
  const std::array<float, o2::track::kNParams>& getCrudeAbsDiffCut() const { return mCrudeAbsDiffCut; }

  ///< set cuts on difference^2/sig^2 of ITS vs TPC track parameters
  void setCrudeNSigma2Cut(const std::array<float, o2::track::kNParams>& vals) { mCrudeNSigma2Cut = vals; }
  ///< get cuts on absolute difference of ITS vs TPC track parameters
  const std::array<float, o2::track::kNParams>& getCrudeNSigma2Cut() const { return mCrudeNSigma2Cut; }

  ///< set cut matching chi2
  void setCutMatchingChi2(float val) { mCutMatchingChi2 = val; }
  ///< get cut on matching chi2
  float getCutMatchingChi2() const { return mCutMatchingChi2; }

  ///< set max number of matching candidates to consider
  void setMaxMatchCandidates(int n) { mMaxMatchCandidates = n > 1 ? 1 : n; }
  ///< get max number of matching candidates to consider
  int getMaxMatchCandidates() const { return mMaxMatchCandidates; }

  ///< set tolerance (TPC time bins) on ITS-TPC times comparison
  void setTimeBinTolerance(float val) { mTimeBinTolerance = val; }
  ///< get tolerance (TPC time bins) on ITS-TPC times comparison
  float getTimeBinTolerance() const { return mTimeBinTolerance; }

  ///< set tolerance on TPC time-bins estimate from highest cluster Z
  void setTPCTimeEdgeZSafeMargin(float val) { mTPCTimeEdgeZSafeMargin = val; }
  ///< get tolerance on TPC time-bins estimate from highest cluster Z
  float getTPCTimeEdgeZSafeMargin() const { return mTPCTimeEdgeZSafeMargin; }

//<<< ====================== cuts ================================<<<

#ifdef _ALLOW_DEBUG_TREES_
  enum DebugFlagTypes : UInt_t {
    MatchTreeAll = 0x1 << 1,     ///< produce matching candidates tree for all candidates
    MatchTreeAccOnly = 0x1 << 2, ///< fill the matching candidates tree only once the cut is passed
    WinnerMatchesTree = 0x1 << 3 ///< separate debug tree for winner matches
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

  ///< get the name of output debug file
  const std::string& getDebugTreeFileName() const { return mDebugTreeFileName; }

  ///< fill matching debug tree
  void fillTPCITSmatchTree(int itsID, int tpcID, int rejFlag, float chi2 = -1.);
  void dumpWinnerMatches();
#endif

 private:
  void assertDPLIO(bool v);
  void attachInputTrees();
  bool prepareTPCTracks();
  bool prepareITSTracks();
  bool prepareFITInfo();
  bool loadTPCTracksNextChunk();
  bool loadITSTracksNextChunk();
  void loadITSClustersChunk(int chunk);
  void loadITSTracksChunk(int chunk);
  void loadTPCClustersChunk(int chunk);
  void loadTPCTracksChunk(int chunk);

  void doMatching(int sec);

  void refitWinners();
  bool refitTrackTPCITS(int iITS);
  void selectBestMatches();
  void buildMatch2TrackTables();
  bool validateTPCMatch(int mtID);
  void removeITSfromTPC(int itsMatchID, int tpcMatchID);
  void removeTPCfromITS(int tpcMatchID, int itsMatchID);
  bool isValidatedTPC(const matchCand& m);
  bool isValidatedITS(const matchCand& m);
  bool isDisabledTPC(const matchCand& m);
  bool isDisabledITS(const matchCand& m);

  int compareTPCITSTracks(const TrackLocITS& tITS, const TrackLocTPC& tTPC, float& chi2) const;
  float getPredictedChi2NoZ(const o2::track::TrackParCov& tr1, const o2::track::TrackParCov& tr2) const;
  bool propagateToRefX(o2::track::TrackParCov& trc);
  void addTrackCloneForNeighbourSector(const TrackLocITS& src, int sector);

  ///------------------- manipulations with matches records ----------------------
  bool registerMatchRecordTPC(TrackLocITS& tITS, TrackLocTPC& tTPC, float chi2);
  void registerMatchRecordITS(TrackLocITS& tITS, int matchTPCID, float chi2);
  void suppressMatchRecordITS(int matchITSID, int matchTPCID);
  matchCand& getTPCMatchEntry(TrackLocTPC& tTPC);
  matchCand& getITSMatchEntry(TrackLocITS& tITS);

  ///< get number of matching records for TPC track referring to this matchCand
  int getNMatchRecordsTPC(const matchCand& tpcMatch) const;

  ///< get number of matching records for ITS track referring to this matchCand
  int getNMatchRecordsITS(const matchCand& itsMatch) const;

  ///< get number of matching records for TPC track referring to matchTPS struct with matchTPCID
  int getNMatchRecordsTPC(int matchTPCID) const
  {
    return matchTPCID < 0 ? 0 : getNMatchRecordsTPC(mMatchesTPC[matchTPCID]);
  }
  ///< get number of matching records for ITS track referring to matchCand struct with matchITSID
  int getNMatchRecordsITS(int matchITSID) const
  {
    return matchITSID < 0 ? 0 : getNMatchRecordsITS(mMatchesITS[matchITSID]);
  }

  ///< convert TPC time bin to ITS ROFrame units
  int tpcTimeBin2ITSROFrame(float tbin) const
  {
    if (mITSTriggered) {
      return mITSROFofTPCBin[int(tbin > 0 ? tbin : 0)];
    }
    int rof = tbin * mTPCBin2ITSROFrame - mITSROFramePhaseOffset;
    return rof < 0 ? 0 : rof;
  }

  ///< convert ITS ROFrame to TPC time bin units // TOREMOVE
  float itsROFrame2TPCTimeBin(int rof) const { return (rof + mITSROFramePhaseOffset) * mITSROFrame2TPCBin; }

  ///< convert Interaction Record for TPC time bin units
  float intRecord2TPCTimeBin(const o2::InteractionRecord& bc) const
  {
    return bc.differenceInBC(mStartIR) * o2::constants::lhc::LHCBunchSpacingNS / 1000 / mTPCTBinMUS;
  }

  ///< convert Z interval to TPC time-bins
  float z2TPCBin(float z) const { return z * mZ2TPCBin; }

  ///< convert TPC time-bins to Z interval
  float tpcBin2Z(float t) const { return t * mTPCBin2Z; }

  ///< rought check of 2 track params difference, return -1,0,1 if it is <,within or > than tolerance
  int roughCheckDif(float delta, float toler, int rejFlag) const
  {
    return delta > toler ? rejFlag : (delta < -toler ? -rejFlag : Accept);
  }
  //================================================================

  bool mInitDone = false; ///< flag init already done
  bool mDPLIO = false;    ///< inputs are set by from DLP device rather than trees

  int mCurrTPCTracksTreeEntry = -1;   ///< current TPC tracks tree entry loaded to memory
  int mCurrTPCClustersTreeEntry = -1; ///< current TPC clusters tree entry loaded to memory
  int mCurrITSClustersTreeEntry = -1; ///< current ITS clusters tree entry loaded to memory
  int mCurrITSTracksTreeEntry = -1;   ///< current ITS tracks tree entry loaded to memory

  bool mMCTruthON = false; ///< flag availability of MC truth
  o2::InteractionRecord mStartIR; ///< IR corresponding to the start of the TF
  ///========== Parameters to be set externally, e.g. from CCDB ====================

  bool mITSTriggered = false; ///< ITS readout is triggered

  ///< do we use track Z difference to reject fake matches? makes sense for triggered mode only
  bool mCompareTracksDZ = false;

  ///<tolerance on abs. different of ITS/TPC params
  std::array<float, o2::track::kNParams> mCrudeAbsDiffCut = { 2.f, 2.f, 0.2f, 0.2f, 4.f };

  ///<tolerance on per-component ITS/TPC params NSigma
  std::array<float, o2::track::kNParams> mCrudeNSigma2Cut = { 49.f, 49.f, 49.f, 49.f, 49.f };

  float mCutMatchingChi2 = 200.f; ///< cut on matching chi2

  float mSectEdgeMargin2 = 0.; ///< crude check if ITS track should be matched also in neighbouring sector

  int mMaxMatchCandidates = 5; ///< max allowed matching candidates per TPC track

  ///< safety margin (in TPC time bins) for ITS-TPC tracks time (in TPC time bins!) comparison
  float mTPCITSTimeBinSafeMargin = 1.f;

  ///< safety margin in cm when estimating TPC track tMin and tMax from assigned time0 and its
  ///< track Z position
  float mTPCTimeEdgeZSafeMargin = 20.f;

  ///< safety margin in TPC time bins when estimating TPC track tMin and tMax from
  ///< assigned time0 and its track Z position (converted from mTPCTimeEdgeZSafeMargin)
  float mTPCTimeEdgeTSafeMargin = 0.f;
  float mTimeBinTolerance = 10.f; ///<tolerance in time-bin for ITS-TPC time bracket matching

  float mITSROFrameLengthMUS = -1.; ///< ITS RO frame in \mus
  float mITSROFrameOffsetMUS = 0;   ///< time in \mus corresponding to start of 1st ITS ROFrame,
                                    ///< i.e. t = ROFrameID*mITSROFrameLengthMUS - mITSROFrameOffsetMUS
  float mITSROFramePhaseOffset = 0; ///< mITSROFrameOffsetMUS recalculated in mITSROFrameLengthMUS units
  float mTPCVDrift0 = -1.;          ///< TPC nominal drift speed in cm/microseconds
  float mTPCVDrift0Inv = -1.;       ///< TPC nominal drift speed in cm/microseconds
  float mTPCTBinMUS = 0.;           ///< TPC time bin duration in microseconds
  float mITSROFrame2TPCBin = 0.;    ///< conversion coeff from ITS ROFrame units to TPC time-bin
  float mTPCBin2ITSROFrame = 0.;    ///< conversion coeff from TPC time-bin to ITS ROFrame units
  float mZ2TPCBin = 0.;             ///< conversion coeff from Z to TPC time-bin
  float mTPCBin2Z = 0.;             ///< conversion coeff from TPC time-bin to Z
  float mNTPCBinsFullDrift = 0.;    ///< max time bin for full drift
  float mTPCZMax = 0.;              ///< max drift length

  TTree* mTreeITSTracks = nullptr;   ///< input tree for ITS tracks
  TTree* mTreeITSTrackROFRec = nullptr; ///< input tree for ITS Tracks ROFRecords vector
  TTree* mTreeTPCTracks = nullptr;   ///< input tree for TPC tracks
  TTree* mTreeITSClusters = nullptr; ///< input tree for ITS clusters
  TTree* mTreeITSClusterROFRec = nullptr; ///< input tree for ITS Clusters ROFRecords vector
  TTree* mTreeFITInfo = nullptr;     ///< input tree for FIT info

  o2::tpc::ClusterNativeHelper::Reader* mTPCClusterReader = nullptr; ///< TPC cluster reader
  std::unique_ptr<o2::tpc::ClusterNativeAccess> mTPCClusterIdxStructOwn; ///< used in case of tree-based IO
  std::unique_ptr<o2::tpc::ClusterNative[]> mTPCClusterBufferOwn;        ///< buffer for clusters in mTPCClusterIdxStructOwn
  o2::tpc::MCLabelContainer mTPCClusterMCBufferOwn;                      ///< buffer for mc labels

  std::unique_ptr<TPCTransform> mTPCTransform;                       ///< TPC cluster transformation
  std::unique_ptr<o2::gpu::GPUParam> mTPCClusterParam;               ///< TPC clusters error param

  TTree* mOutputTree = nullptr; ///< output tree for matched tracks

  ///>>>------ these are input arrays which should not be modified by the matching code
  //           since this info is provided by external device
  const std::vector<o2::itsmft::ROFRecord>* mITSTrackROFRec = nullptr;       ///< input ITS tracks ROFRecord
  const std::vector<o2::its::TrackITS>* mITSTracksArrayInp = nullptr;        ///< input ITS tracks
  const std::vector<o2::tpc::TrackTPC>* mTPCTracksArrayInp = nullptr;        ///< input TPC tracks
  gsl::span<const int> mITSTrackClusIdxSPAN;                                 ///< input ITS track cluster indices span from DPL
  const std::vector<int>* mITSTrackClusIdxInp = nullptr;                     ///< input ITS track cluster indices
  const std::vector<o2::itsmft::Cluster>* mITSClustersArrayInp = nullptr;    ///< input ITS clusters
  const std::vector<o2::itsmft::ROFRecord>* mITSClusterROFRec = nullptr;     ///< input ITS clusters ROFRecord
  const std::vector<o2::ft0::RecPoints>* mFITInfoInp = nullptr;              ///< optional input FIT info
  const o2::tpc::ClusterNativeAccess* mTPCClusterIdxStruct = nullptr;        ///< struct holding the TPC cluster indices

  const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mITSTrkLabels = nullptr; ///< input ITS Track MC labels
  const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mTPCTrkLabels = nullptr; ///< input TPC Track MC labels
  /// <<<-----

  ///< container for matchCand structures of TPC tracks (1 per TPCtrack with some matches to ITS)
  std::vector<matchCand> mMatchesTPC;
  ///< container for matchCand structures of ITS tracks(1 per ITStrack with some matches to TPC)
  std::vector<matchCand> mMatchesITS;

  ///< container for record the match of TPC track to single ITS track
  std::vector<matchRecord> mMatchRecordsTPC;
  ///< container for reference to matchRecord involving particular ITS track
  std::vector<matchRecord> mMatchRecordsITS;

  ///< track in mITSWork have pointer on matches in mMatchesITS, but not vice versa
  ///< here we will keep index of ITS track in mITSWork for each match
  std::vector<int> mITSMatch2Track;

  ///< track in mTPCWork have pointer on matches in mMatchesTPC, but not vice versa
  ///< here we will keep index of TPC track in mTPCWork for each match
  std::vector<int> mTPCMatch2Track;

  std::vector<int> mITSROFofTPCBin;         ///< aux structure for mapping of TPC time-bins on ITS ROFs
  std::vector<TimeBracket> mITSROFTimes;    ///< min/max times of ITS ROFs in TPC time-bins
  std::vector<TrackLocTPC> mTPCWork;        ///< TPC track params prepared for matching
  std::vector<TrackLocITS> mITSWork;        ///< ITS track params prepared for matching
  std::vector<o2::MCCompLabel> mTPCLblWork; ///< TPC track labels
  std::vector<o2::MCCompLabel> mITSLblWork; ///< ITS track labels
  std::vector<float> mWinnerChi2Refit;      ///< vector of refitChi2 for winners

  ///< per sector indices of TPC track entry in mTPCWork
  std::array<std::vector<int>, o2::constants::math::NSectors> mTPCSectIndexCache;
  ///< per sector indices of ITS track entry in mITSWork
  std::array<std::vector<int>, o2::constants::math::NSectors> mITSSectIndexCache;

  ///<indices of 1st entries with time-bin above the value
  std::array<std::vector<int>, o2::constants::math::NSectors> mTPCTimeBinStart;
  ///<indices of 1st entries of ITS tracks with givem ROframe
  std::array<std::vector<int>, o2::constants::math::NSectors> mITSTimeBinStart;
  ///<outputs tracks container
  std::vector<o2::dataformats::TrackTPCITS> mMatchedTracks;
  std::vector<o2::MCCompLabel> mOutITSLabels; ///< ITS label of matched track
  std::vector<o2::MCCompLabel> mOutTPCLabels; ///< TPC label of matched track

  std::string mITSTrackBranchName = "ITSTrack";          ///< name of branch containing input ITS tracks
  std::string mITSTrackClusIdxBranchName = "ITSTrackClusIdx"; ///< name of branch containing input ITS tracks cluster indices
  std::string mITSTrackROFRecBranchName = "ITSTracksROF"; ///< name of branch containing input ITS tracks ROFRecords
  std::string mTPCTrackBranchName = "Tracks";            ///< name of branch containing input TPC tracks
  std::string mITSClusterBranchName = "ITSCluster";      ///< name of branch containing input ITS clusters
  std::string mITSClusterROFRecBranchName = "ITSClustersROF"; ///< name of branch containing input ITS clusters ROFRecords
  std::string mITSMCTruthBranchName = "ITSTrackMCTruth"; ///< name of branch containing ITS MC labels
  std::string mTPCMCTruthBranchName = "TracksMCTruth";   ///< name of branch containing input TPC tracks
  std::string mFITInfoBranchName = "FT0Cluster";         ///< name of branch containing input FIT Info
  std::string mOutTPCITSTracksBranchName = "TPCITS";     ///< name of branch containing output matched tracks
  std::string mOutTPCMCTruthBranchName = "MatchTPCMCTruth"; ///< name of branch for output matched tracks TPC MC
  std::string mOutITSMCTruthBranchName = "MatchITSMCTruth"; ///< name of branch for output matched tracks ITS MC

#ifdef _ALLOW_DEBUG_TREES_
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  UInt_t mDBGFlags = 0;
  std::string mDebugTreeFileName = "dbg_TPCITSmatch.root"; ///< name for the debug tree file
#endif

  ///----------- aux stuff --------------///
  static constexpr float Tan70 = 2.74747771e+00;       // tg(70 degree): std::tan(70.*o2::constants::math::PI/180.);
  static constexpr float Cos70I2 = 1. + Tan70 * Tan70; // 1/cos^2(70) = 1 + tan^2(70)
  static constexpr float MaxSnp = 0.9;                 // max snp of ITS or TPC track at xRef to be matched
  static constexpr float MaxTgp = 2.064;               // max tg corresponting to MaxSnp = MaxSnp/std::sqrt(1.-MaxSnp^2)

  TStopwatch mTimerTot;
  TStopwatch mTimerIO;
  TStopwatch mTimerDBG;
  TStopwatch mTimerReg;
  TStopwatch mTimerRefit;

  ClassDefNV(MatchTPCITS, 1);
};

//______________________________________________
inline matchCand& MatchTPCITS::getTPCMatchEntry(TrackLocTPC& tTPC)
{
  ///< return the matchCand entry referred by the tTPC track,
  ///< create if neaded
  if (tTPC.matchID == MinusOne) { // does this TPC track already have any match? If not, create matchCand entry
    tTPC.matchID = mMatchesTPC.size();
    mMatchesTPC.emplace_back(tTPC.source);
    return mMatchesTPC.back();
  }
  return mMatchesTPC[tTPC.matchID];
}

//______________________________________________
inline matchCand& MatchTPCITS::getITSMatchEntry(TrackLocITS& tITS)
{
  ///< return the matchCand entry referred by the tITS track,
  ///< create if neaded
  if (tITS.matchID == MinusOne) { // does this ITS track already have any match? If not, create matchCand entry
    tITS.matchID = mMatchesITS.size();
    mMatchesITS.emplace_back(tITS.source);
    return mMatchesITS.back();
  }
  return mMatchesITS[tITS.matchID];
}

//______________________________________________
inline bool MatchTPCITS::isValidatedTPC(const matchCand& m)
{
  return m.first > MinusOne && mMatchRecordsTPC[m.first].nextRecID == Validated;
}

//______________________________________________
inline bool MatchTPCITS::isValidatedITS(const matchCand& m)
{
  return m.first > MinusOne && mMatchRecordsITS[m.first].nextRecID == Validated;
}

//______________________________________________
inline bool MatchTPCITS::isDisabledITS(const matchCand& m) { return m.first < 0; }

//______________________________________________
inline bool MatchTPCITS::isDisabledTPC(const matchCand& m) { return m.first < 0; }

//______________________________________________
inline void MatchTPCITS::removeTPCfromITS(int tpcMatchID, int itsMatchID)
{
  ///< remove reference to TPC match tpcMatchID from ITS match itsMatchID
  auto& itsMatch = mMatchesITS[itsMatchID];
  if (isValidatedITS(itsMatch))
    return;
  int topID = MinusOne, next = itsMatch.first;
  while (next > MinusOne) {
    auto& rcITS = mMatchRecordsITS[next];
    if (rcITS.matchID == tpcMatchID) {
      if (topID < 0) {
        itsMatch.first = rcITS.nextRecID;
      } else {
        mMatchRecordsITS[topID].nextRecID = rcITS.nextRecID;
      }
      return;
    }
    topID = next;
    next = rcITS.nextRecID;
  }
}

//______________________________________________
inline void MatchTPCITS::removeITSfromTPC(int itsMatchID, int tpcMatchID)
{
  ///< remove reference to ITS match itsMatchID from TPC match tpcMatchID
  auto& tpcMatch = mMatchesTPC[tpcMatchID];
  if (isValidatedTPC(tpcMatch))
    return;
  int topID = MinusOne, next = tpcMatch.first;
  while (next > MinusOne) {
    auto& rcTPC = mMatchRecordsTPC[next];
    if (rcTPC.matchID == itsMatchID) {
      if (topID < 0) {
        tpcMatch.first = rcTPC.nextRecID;
      } else {
        mMatchRecordsTPC[topID].nextRecID = rcTPC.nextRecID;
      }
      return;
    }
    topID = next;
    next = rcTPC.nextRecID;
  }
}
}
}

#endif
