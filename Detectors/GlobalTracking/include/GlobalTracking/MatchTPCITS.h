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
#include <deque>
#include <vector>
#include <string>
#include <TStopwatch.h>
#include "DataFormatsTPC/TrackTPC.h"
#include "DetectorsBase/Propagator.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "MathUtils/Bracket.h"
#include "CommonDataFormat/EvIndex.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsFT0/RecPoints.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "ITSReconstruction/RecoGeomHelper.h"
#include "TPCFastTransform.h"

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

namespace gpu
{
struct GPUParam;
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

///< TPC track parameters propagated to reference X, with time bracket and index of
///< original track in the currently loaded TPC reco output
struct TrackLocTPC : public o2::track::TrackParCov {
  o2::dataformats::EvIndex<int, int> source; ///< track origin id
  o2::utils::Bracket<float> timeBins;        ///< bracketing time-bins
  float zMin = 0;                            // min possible Z of this track
  float zMax = 0;                            // max possible Z of this track
  int matchID = MinusOne;                    ///< entry (non if MinusOne) of best matchRecord in mMatchRecordsTPC
  TrackLocTPC(const o2::track::TrackParCov& src, int tch, int tid) : o2::track::TrackParCov(src), source(tch, tid) {}
  TrackLocTPC() = default;
  ClassDefNV(TrackLocTPC, 1);
};

///< ITS track outward parameters propagated to reference X, with time bracket and index of
///< original track in the currently loaded ITS reco output
struct TrackLocITS : public o2::track::TrackParCov {
  o2::dataformats::EvIndex<int, int> source; ///< track origin id
  int roFrame = MinusOne;                    ///< ITS readout frame assigned to this track
  int matchID = MinusOne;                    ///< entry (non if MinusOne) of best matchRecord in mMatchRecordsITS
  TrackLocITS(const o2::track::TrackParCov& src, int tch, int tid) : o2::track::TrackParCov(src), source(tch, tid) {}
  TrackLocITS() = default;
  ClassDefNV(TrackLocITS, 1);
};

///< record TPC or ITS track associated with single ITS or TPC track and reference on
///< the next (worse chi2) matchRecord of the same TPC or ITS track
struct matchRecord {
  float chi2 = -1.f;        ///< matching chi2
  int partnerID = MinusOne; ///< id of parnter track entry in mTPCWork or mITSWork containers
  int nextRecID = MinusOne; ///< index of eventual next record

  matchRecord(int partID, float chi2match) : partnerID(partID), chi2(chi2match) {}
  matchRecord(int partID, float chi2match, int nxt) : partnerID(partID), chi2(chi2match), nextRecID(nxt) {}
  matchRecord() = default;
};

///< Link of the AfterBurner track: update at sertain cluster
///< original track in the currently loaded TPC reco output
struct ABTrackLink : public o2::track::TrackParCov {
  static constexpr int Disabled = -2;
  int clID = MinusOne;     ///< ID of the attached cluster
  int parentID = MinusOne; ///< ID of the parent link (on prev layer) or parent TPC seed
  int nextOnLr = MinusOne; ///< ID of the next (in quality) link on the same layer
  int icCandID = MinusOne; ///< ID of the interaction candidate this track belongs to
  float chi2 = 0.f;        ///< chi2 after update
  ABTrackLink() = default;
  ABTrackLink(const o2::track::TrackParCov& src, int ic, int parid = -1, int clid = -1) : o2::track::TrackParCov(src), clID(clid), parentID(parid), icCandID(ic) {}
  bool isDisabled() const { return clID == Disabled; }
  void disable() { clID = Disabled; }
};

struct ABTrackLinksList {
  int trackID = MinusOne; ///< TPC work track id
  std::array<int, o2::its::RecoGeomHelper::getNLayers() + 1> firstInLr;
  ABTrackLinksList(int id = MinusOne) : trackID(id)
  {
    firstInLr.fill(MinusOne);
  }
};

struct InteractionCandidate : public o2::InteractionRecord {
  o2::utils::Bracket<float> timeBins; // interaction time (int TPC time bins)
  int rofITS;                         // corresponding ITS ROF entry (in the ROFRecord vectors)
  uint32_t flag;                      // origin, etc.
  void* clRefPtr = nullptr;           // pointer on cluster references container (if any)
  InteractionCandidate() = default;
  InteractionCandidate(const o2::InteractionRecord& ir, float t, float dt, int rof, uint32_t f = 0) : o2::InteractionRecord(ir), timeBins(t - dt, t + dt), rofITS(rof), flag(f) {}
};

struct ITSChipClustersRefs {
  ///< contaner for sorted cluster indices for certain time window (usually ROF) and reference on the start and N clusters
  ///< for every chip
  using ClusRange = o2::dataformats::RangeReference<int, int>;
  std::vector<int> clusterID;                                           // indices of sorted clusters
  std::array<ClusRange, o2::its::RecoGeomHelper::getNChips()> chipRefs; // offset and number of clusters in each chip
  ITSChipClustersRefs(int nclIni = 50000)
  {
    clusterID.reserve(nclIni);
  }
  void clear()
  {
    clusterID.clear();
    std::memset(chipRefs.data(), 0, chipRefs.size() * sizeof(ClusRange)); // reset chip->cluster references
  }
};

class MatchTPCITS
{
  using ClusRange = o2::dataformats::RangeReference<int, int>;
  using MCLabCont = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  using TPCTransform = o2::gpu::TPCFastTransform;
  using BracketF = o2::utils::Bracket<float>;

 public:
  MatchTPCITS(); // std::unique_ptr to forward declared type needs constructor / destructor in .cxx
  ~MatchTPCITS();

  static constexpr float XTPCInnerRef = 83.0;                            ///< reference radius at which TPC provides the tracks
  static constexpr float XTPCOuterRef = 255.0;                           ///< reference radius to propagate outer TPC track
  static constexpr float XMatchingRef = 70.0;                            ///< reference radius to propage tracks for matching
  static constexpr float YMaxAtXMatchingRef = XMatchingRef * 0.17632698; ///< max Y in the sector at reference X

  static constexpr int MaxUpDnLadders = 3;                     // max N ladders to check up and down from selected one
  static constexpr int MaxLadderCand = 2 * MaxUpDnLadders + 1; // max ladders to check for matching clusters
  static constexpr int MaxSeedsPerLayer = 50;                  // TODO
  static constexpr int NITSLayers = o2::its::RecoGeomHelper::getNLayers();
  ///< perform matching for provided input
  void run();

  // RSTODO
  void runAfterBurner();
  bool runAfterBurner(int tpcWID, int iCStart, int iCEnd);
  float correctTPCTrack(o2::track::TrackParCov& trc, const TrackLocTPC& tTPC, const InteractionCandidate& cand) const;
  bool checkABSeedFromLr(int lrSeed, int seedID, ABTrackLinksList& llist, int& nMissed);
  ABTrackLinksList& getCreateABTrackLinksList(int tpcWID);
  int registerABLink(ABTrackLinksList& llist, const o2::track::TrackParCov& src, int ic, int lr, int parentID = -1, int clID = -1);
  void printABTree(const ABTrackLinksList& llist) const;
  bool isBetter(const o2::track::TrackParCov& src, const ABTrackLink& lnk) { return true; } // RS TODO

  int prepareInteractionTimes();

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

  ///< After-burner getter/setter
  void setRunAfterBurner(bool v) { mRunAfterBurner = v; }
  bool isRunAfterBurner() const { return mRunAfterBurner; }

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
  void setITSTrkLabelsInp(const MCLabCont* lbl)
  {
    assertDPLIO(true);
    mITSTrkLabels = lbl;
  }

  ///< set input ITS clusters MC labels received via DPL
  void setITSClsLabelsInp(const MCLabCont* lbl)
  {
    assertDPLIO(true);
    mITSClsLabels = lbl;
  }

  ///< set input TPC track MC labels received via DPL
  void setTPCTrkLabelsInp(const MCLabCont* lbl)
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
  void setITSClusMCTruthBranchName(const std::string& nm) { mITSClusMCTruthBranchName = nm; }
  void setTPCMCTruthBranchName(const std::string& nm) { mTPCMCTruthBranchName = nm; }
  void setFITInfoBbranchName(const std::string& nm) { mFITInfoBranchName = nm; }
  void setOutTPCITSTracksBranchName(const std::string& nm) { mOutTPCITSTracksBranchName = nm; }
  void setOutTPCMCTruthBranchName(const std::string& nm) { mOutTPCMCTruthBranchName = nm; }
  void setOutITSMCTruthBranchName(const std::string& nm) { mOutITSMCTruthBranchName = nm; }

  ///< get input branch names for the input from the tree
  const std::string& getITSTrackBranchName() const { return mITSTrackBranchName; }
  const std::string& getTPCTrackBranchName() const { return mTPCTrackBranchName; }
  const std::string& getITSClusterBranchName() const { return mITSClusterBranchName; }
  const std::string& getITSClusMCTruthBranchName() const { return mITSClusMCTruthBranchName; }
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

  //>>> ====================== options =============================>>>
  void setUseMatCorrFlag(int f);
  int getUseMatCorrFlag() const { return mUseMatCorrFlag; }

  //<<< ====================== options =============================<<<

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

  ///< set cut on AB track-cluster chi2
  void setCutABTrack2ClChi2(float val) { mCutABTrack2ClChi2 = val; }
  ///< get cut on matching chi2
  float getCutABTrack2ClChi2() const { return mCutABTrack2ClChi2; }

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
  int findLaddersToCheckBOn(int ilr, int lad0, const o2::utils::CircleXY& circle, float errYFrac,
                            std::array<int, MaxLadderCand>& lad2Check) const;
  int findLaddersToCheckBOff(int ilr, int lad0, const o2::utils::IntervalXY& trcLinPar, float errYFrac,
                             std::array<int, MatchTPCITS::MaxLadderCand>& lad2Check) const;

  void assertDPLIO(bool v);
  void attachInputTrees();
  int prepareTPCTracksAfterBurner();
  bool prepareTPCTracks();
  bool prepareITSTracks();
  bool prepareITSClusters();
  bool prepareFITInfo();
  bool loadTPCTracksNextChunk();
  bool loadITSTracksNextChunk();
  void loadITSClustersChunk(int chunk);
  void loadITSTracksChunk(int chunk);
  void loadTPCClustersChunk(int chunk);
  void loadTPCTracksChunk(int chunk);

  int preselectChipClusters(std::vector<int>& clVecOut, const ClusRange& clRange, const ITSChipClustersRefs& clRefs,
                            float trackY, float trackZ, float tolerY, float tolerZ,
                            const o2::MCCompLabel& lblTrc) const;
  void fillClustersForAfterBurner(ITSChipClustersRefs& refCont, int rofStart, int nROFs = 1);
  void cleanAfterBurnerClusRefCache(int currentIC, int& startIC);
  void flagUsedITSClusters(const o2::its::TrackITS& track, int rofOffset);

  void doMatching(int sec);

  void refitWinners(bool loopInITS = false);
  bool refitTrackTPCITSloopITS(int iITS, int& iTPC);
  bool refitTrackTPCITSloopTPC(int iTPC, int& iITS);
  void selectBestMatches();
  bool validateTPCMatch(int iTPC);
  void removeITSfromTPC(int itsID, int tpcID);
  void removeTPCfromITS(int tpcID, int itsID);
  bool isValidatedTPC(const TrackLocTPC& t) const;
  bool isValidatedITS(const TrackLocITS& t) const;
  bool isDisabledTPC(const TrackLocTPC& t) const;
  bool isDisabledITS(const TrackLocITS& t) const;

  int compareTPCITSTracks(const TrackLocITS& tITS, const TrackLocTPC& tTPC, float& chi2) const;
  float getPredictedChi2NoZ(const o2::track::TrackParCov& tr1, const o2::track::TrackParCov& tr2) const;
  bool propagateToRefX(o2::track::TrackParCov& trc);
  void addTrackCloneForNeighbourSector(const TrackLocITS& src, int sector);

  ///------------------- manipulations with matches records ----------------------
  bool registerMatchRecordTPC(int iITS, int iTPC, float chi2);
  void registerMatchRecordITS(int iITS, int iTPC, float chi2);
  void suppressMatchRecordITS(int iITS, int iTPC);

  ///< get number of matching records for TPC track
  int getNMatchRecordsTPC(const TrackLocTPC& tTPC) const;

  ///< get number of matching records for ITS track
  int getNMatchRecordsITS(const TrackLocITS& tITS) const;

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
  bool mFieldON = true;   ///< flag for field ON/OFF

  int mCurrTPCTracksTreeEntry = -1;   ///< current TPC tracks tree entry loaded to memory
  int mCurrTPCClustersTreeEntry = -1; ///< current TPC clusters tree entry loaded to memory
  int mCurrITSTracksTreeEntry = -1;   ///< current ITS tracks tree entry loaded to memory

  bool mMCTruthON = true;         ///< flag availability of MC truth
  o2::InteractionRecord mStartIR; ///< IR corresponding to the start of the TF
  ///========== Parameters to be set externally, e.g. from CCDB ====================
  int mUseMatCorrFlag = o2::base::Propagator::USEMatCorrTGeo;

  bool mITSTriggered = false; ///< ITS readout is triggered

  ///< do we use track Z difference to reject fake matches? makes sense for triggered mode only
  bool mCompareTracksDZ = false;

  ///< do we want to run an afterburner for secondaries matchint?
  bool mRunAfterBurner = true;

  ///<tolerance on abs. different of ITS/TPC params
  std::array<float, o2::track::kNParams> mCrudeAbsDiffCut = {2.f, 2.f, 0.2f, 0.2f, 4.f};

  ///<tolerance on per-component ITS/TPC params NSigma
  std::array<float, o2::track::kNParams> mCrudeNSigma2Cut = {49.f, 49.f, 49.f, 49.f, 49.f};

  float mCutMatchingChi2 = 200.f; ///< cut on matching chi2

  float mCutABTrack2ClChi2 = 20.f; ///< cut on AfterBurner track-cluster chi2

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
  float mTPCVDrift0Inv = -1.;       ///< inverse TPC nominal drift speed in cm/microseconds
  float mTPCTBinMUS = 0.;           ///< TPC time bin duration in microseconds
  float mITSROFrame2TPCBin = 0.;    ///< conversion coeff from ITS ROFrame units to TPC time-bin
  float mTPCBin2ITSROFrame = 0.;    ///< conversion coeff from TPC time-bin to ITS ROFrame units
  float mZ2TPCBin = 0.;             ///< conversion coeff from Z to TPC time-bin
  float mTPCBin2Z = 0.;             ///< conversion coeff from TPC time-bin to Z
  float mNTPCBinsFullDrift = 0.;    ///< max time bin for full drift
  float mTPCZMax = 0.;              ///< max drift length

  TTree* mTreeITSTracks = nullptr;        ///< input tree for ITS tracks
  TTree* mTreeITSTrackROFRec = nullptr;   ///< input tree for ITS Tracks ROFRecords vector
  TTree* mTreeTPCTracks = nullptr;        ///< input tree for TPC tracks
  TTree* mTreeITSClusters = nullptr;      ///< input tree for ITS clusters
  TTree* mTreeITSClusterROFRec = nullptr; ///< input tree for ITS Clusters ROFRecords vector
  TTree* mTreeFITInfo = nullptr;          ///< input tree for FIT info

  o2::tpc::ClusterNativeHelper::Reader* mTPCClusterReader = nullptr;     ///< TPC cluster reader
  std::unique_ptr<o2::tpc::ClusterNativeAccess> mTPCClusterIdxStructOwn; ///< used in case of tree-based IO
  std::unique_ptr<o2::tpc::ClusterNative[]> mTPCClusterBufferOwn;        ///< buffer for clusters in mTPCClusterIdxStructOwn
  o2::tpc::MCLabelContainer mTPCClusterMCBufferOwn;                      ///< buffer for mc labels

  std::unique_ptr<TPCTransform> mTPCTransform;         ///< TPC cluster transformation
  std::unique_ptr<o2::gpu::GPUParam> mTPCClusterParam; ///< TPC clusters error param

  TTree* mOutputTree = nullptr; ///< output tree for matched tracks

  ///>>>------ these are input arrays which should not be modified by the matching code
  //           since this info is provided by external device
  const std::vector<o2::itsmft::ROFRecord>* mITSTrackROFRec = nullptr;    ///< input ITS tracks ROFRecord
  const std::vector<o2::its::TrackITS>* mITSTracksArrayInp = nullptr;     ///< input ITS tracks
  const std::vector<o2::tpc::TrackTPC>* mTPCTracksArrayInp = nullptr;     ///< input TPC tracks
  gsl::span<const int> mITSTrackClusIdxSPAN;                              ///< input ITS track cluster indices span from DPL
  const std::vector<int>* mITSTrackClusIdxInp = nullptr;                  ///< input ITS track cluster indices
  const std::vector<o2::itsmft::Cluster>* mITSClustersArrayInp = nullptr; ///< input ITS clusters
  const std::vector<o2::itsmft::ROFRecord>* mITSClusterROFRec = nullptr;  ///< input ITS clusters ROFRecord
  const std::vector<o2::ft0::RecPoints>* mFITInfoInp = nullptr;           ///< optional input FIT info
  const o2::tpc::ClusterNativeAccess* mTPCClusterIdxStruct = nullptr;     ///< struct holding the TPC cluster indices

  const MCLabCont* mITSTrkLabels = nullptr; ///< input ITS Track MC labels
  const MCLabCont* mITSClsLabels = nullptr; ///< input ITS Cluster MC labels
  const MCLabCont* mTPCTrkLabels = nullptr; ///< input TPC Track MC labels
  /// <<<-----
  std::vector<o2::itsmft::Cluster> mITSClustersBuffer; ///< input ITS clusters buffer for tree IO
  std::vector<o2::itsmft::ROFRecord> mITSClusterROFRecBuffer;
  MCLabCont mITSClsLabelsBuffer;

  std::vector<InteractionCandidate> mInteractions; ///< possible interaction times

  ///< container for record the match of TPC track to single ITS track
  std::vector<matchRecord> mMatchRecordsTPC;
  ///< container for reference to matchRecord involving particular ITS track
  std::vector<matchRecord> mMatchRecordsITS;

  std::vector<int> mITSROFofTPCBin;         ///< aux structure for mapping of TPC time-bins on ITS ROFs
  std::vector<BracketF> mITSROFTimes;       ///< min/max times of ITS ROFs in TPC time-bins
  std::vector<TrackLocTPC> mTPCWork;        ///< TPC track params prepared for matching
  std::vector<TrackLocITS> mITSWork;        ///< ITS track params prepared for matching
  std::vector<o2::MCCompLabel> mTPCLblWork; ///< TPC track labels
  std::vector<o2::MCCompLabel> mITSLblWork; ///< ITS track labels
  std::vector<float> mWinnerChi2Refit;      ///< vector of refitChi2 for winners

  std::vector<uint8_t> mITSClustersFlags;               ///< flags for used ITS clusters
  std::deque<ITSChipClustersRefs> mITSChipClustersRefs; ///< range of clusters for each chip in ITS (for AfterBurner)

  std::vector<ABTrackLinksList> mABTrackLinksList; ///< pool ... TODO
  std::vector<ABTrackLink> mABLinks;               ///< pool AB track links
  int mMaxABLinksOnLayer = 20;                     ///< max number of candidate links per layer

  ///< per sector indices of TPC track entry in mTPCWork
  std::array<std::vector<int>, o2::constants::math::NSectors> mTPCSectIndexCache;
  ///< per sector indices of ITS track entry in mITSWork
  std::array<std::vector<int>, o2::constants::math::NSectors> mITSSectIndexCache;

  ///< indices of selected track entries in mTPCWork (for tracks selected by AfterBurner)
  std::vector<int> mTPCABIndexCache;
  ///< indices of 1st entries with time-bin above the value
  std::vector<int> mTPCABTimeBinStart;

  ///< indices of 1st entries with time-bin above the value
  std::array<std::vector<int>, o2::constants::math::NSectors> mTPCTimeBinStart;
  ///< indices of 1st entries of ITS tracks with givem ROframe
  std::array<std::vector<int>, o2::constants::math::NSectors> mITSTimeBinStart;
  ///< outputs tracks container
  std::vector<o2::dataformats::TrackTPCITS> mMatchedTracks;
  std::vector<o2::MCCompLabel> mOutITSLabels; ///< ITS label of matched track
  std::vector<o2::MCCompLabel> mOutTPCLabels; ///< TPC label of matched track

  std::string mITSTrackBranchName = "ITSTrack";               ///< name of branch containing input ITS tracks
  std::string mITSTrackClusIdxBranchName = "ITSTrackClusIdx"; ///< name of branch containing input ITS tracks cluster indices
  std::string mITSTrackROFRecBranchName = "ITSTracksROF";     ///< name of branch containing input ITS tracks ROFRecords
  std::string mTPCTrackBranchName = "Tracks";                 ///< name of branch containing input TPC tracks
  std::string mITSClusterBranchName = "ITSCluster";           ///< name of branch containing input ITS clusters
  std::string mITSClusMCTruthBranchName = "ITSClusterMCTruth";///< name of branch containing input ITS clusters MC
  std::string mITSClusterROFRecBranchName = "ITSClustersROF"; ///< name of branch containing input ITS clusters ROFRecords
  std::string mITSMCTruthBranchName = "ITSTrackMCTruth";      ///< name of branch containing ITS MC labels
  std::string mTPCMCTruthBranchName = "TracksMCTruth";        ///< name of branch containing input TPC tracks
  std::string mFITInfoBranchName = "FT0Cluster";              ///< name of branch containing input FIT Info
  std::string mOutTPCITSTracksBranchName = "TPCITS";          ///< name of branch containing output matched tracks
  std::string mOutTPCMCTruthBranchName = "MatchTPCMCTruth";   ///< name of branch for output matched tracks TPC MC
  std::string mOutITSMCTruthBranchName = "MatchITSMCTruth";   ///< name of branch for output matched tracks ITS MC

  o2::its::RecoGeomHelper mRGHelper; ///< helper for cluster and geometry access
  float mITSFiducialZCut = 9999.;    ///< eliminate TPC seeds outside of this range

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
  static constexpr float MinTBToCleanCache = 600.;     // keep in AB ITS cluster refs cache at most this number of TPC bins

  TStopwatch mTimerTot;
  TStopwatch mTimerIO;
  TStopwatch mTimerDBG;
  TStopwatch mTimerRefit;

  ClassDefNV(MatchTPCITS, 1);
};

//______________________________________________
inline bool MatchTPCITS::isValidatedTPC(const TrackLocTPC& t) const
{
  return t.matchID > MinusOne && mMatchRecordsTPC[t.matchID].nextRecID == Validated;
}

//______________________________________________
inline bool MatchTPCITS::isValidatedITS(const TrackLocITS& t) const
{
  return t.matchID > MinusOne && mMatchRecordsITS[t.matchID].nextRecID == Validated;
}

//______________________________________________
inline bool MatchTPCITS::isDisabledITS(const TrackLocITS& t) const { return t.matchID < 0; }

//______________________________________________
inline bool MatchTPCITS::isDisabledTPC(const TrackLocTPC& t) const { return t.matchID < 0; }

//______________________________________________
inline void MatchTPCITS::removeTPCfromITS(int tpcID, int itsID)
{
  ///< remove reference to tpcID track from itsID track matches
  auto& tITS = mITSWork[itsID];
  if (isValidatedITS(tITS)) {
    return;
  }
  int topID = MinusOne, next = tITS.matchID; // ITS matchRecord
  while (next > MinusOne) {
    auto& rcITS = mMatchRecordsITS[next];
    if (rcITS.partnerID == tpcID) {
      if (topID < 0) {
        tITS.matchID = rcITS.nextRecID;
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
inline void MatchTPCITS::removeITSfromTPC(int itsID, int tpcID)
{
  ///< remove reference to itsID track from matches of tpcID track
  auto& tTPC = mTPCWork[tpcID];
  if (isValidatedTPC(tTPC)) {
    return;
  }
  int topID = MinusOne, next = tTPC.matchID;
  while (next > MinusOne) {
    auto& rcTPC = mMatchRecordsTPC[next];
    if (rcTPC.partnerID == itsID) {
      if (topID < 0) {
        tTPC.matchID = rcTPC.nextRecID;
      } else {
        mMatchRecordsTPC[topID].nextRecID = rcTPC.nextRecID;
      }
      return;
    }
    topID = next;
    next = rcTPC.nextRecID;
  }
}
<<<<<<< HEAD
} // namespace globaltracking
} // namespace o2
=======

//______________________________________________
inline void MatchTPCITS::flagUsedITSClusters(const o2::its::TrackITS& track, int rofOffset)
{
  // flag clusters used by this track
  int clEntry = track.getFirstClusterEntry();
  for (int icl = track.getNumberOfClusters(); icl--;) {
    mITSClustersFlags[rofOffset + (*mITSTrackClusIdxInp)[clEntry++]] = 1;
  }
}
//__________________________________________________________
inline int MatchTPCITS::preselectChipClusters(std::vector<int>& clVecOut, const ClusRange& clRange, const ITSChipClustersRefs& clRefs,
                                              float trackY, float trackZ, float tolerY, float tolerZ,
                                              const o2::MCCompLabel& lblTrc) const // TODO lbl is not needed
{
  clVecOut.clear();
  int icID = clRange.getFirstEntry();
  for (int icl = clRange.getEntries(); icl--;) { // note: clusters within a chip are sorted in Z
    int clID = clRefs.clusterID[icID++];         // so, we go in clusterID increasing direction
    const auto& cls = (*mITSClustersArrayInp)[clID];
    float dz = trackZ - cls.getZ();
    auto label = mITSClsLabels->getLabels(clID)[0]; // tmp
    //    if (!(label == lblTrc)) {
    //      continue; // tmp
    //    }
    LOG(INFO) << "cl" << icl << '/' << clID << " " << label
              << " dZ: " << dz << " [" << tolerZ << "| dY: " << trackY - cls.getY() << " [" << tolerY << "]";
    if (dz > tolerZ) {
      float clsZ = cls.getZ();
      LOG(INFO) << "Skip the rest since " << trackZ << " > " << clsZ << "\n";
      break;
    } else if (dz < -tolerZ) {
      LOG(INFO) << "Skip cluster dz=" << dz << " Ztr=" << trackZ << " zCl=" << cls.getZ();
      continue;
    }
    if (fabs(trackY - cls.getY()) > tolerY) {
      LOG(INFO) << "Skip cluster dy= " << trackY - cls.getY() << " Ytr=" << trackY << " yCl=" << cls.getY();
      continue;
    }
    clVecOut.push_back(clID);
  }
  return clVecOut.size();
}

//______________________________________________
inline void MatchTPCITS::cleanAfterBurnerClusRefCache(int currentIC, int& startIC)
{
  // check if some of cached cluster reference from tables startIC to currentIC can be released,
  // they will be necessarily in front slots of the mITSChipClustersRefs
  while (startIC < currentIC && mInteractions[currentIC].timeBins.min() - mInteractions[startIC].timeBins.max() > MinTBToCleanCache) {
    LOG(INFO) << "CAN REMOVE CACHE FOR " << startIC << " curent IC=" << currentIC;
    while (mInteractions[startIC].clRefPtr == &mITSChipClustersRefs.front()) {
      LOG(INFO) << "Reset cache pointer" << mInteractions[startIC].clRefPtr << " for IC=" << startIC;
      mInteractions[startIC++].clRefPtr = nullptr;
    }
    LOG(INFO) << "Reset cache slot " << &mITSChipClustersRefs.front();
    mITSChipClustersRefs.pop_front();
  }
}
}
}
>>>>>>> Matching

#endif
