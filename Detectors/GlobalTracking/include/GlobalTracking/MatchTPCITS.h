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

#define _ALLOW_DEBUG_AB_ // fill extra debug info for AB

#include <Rtypes.h>
#include <array>
#include <deque>
#include <vector>
#include <string>
#include <gsl/span>
#include <TStopwatch.h>
#include "DataFormatsTPC/TrackTPC.h"
#include "DetectorsBase/Propagator.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "MathUtils/Primitive2D.h"
#include "CommonDataFormat/EvIndex.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"
#include "CommonDataFormat/BunchFilling.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsFT0/RecPoints.h"
#include "FT0Reconstruction/InteractionTag.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "ITSReconstruction/RecoGeomHelper.h"
#include "TPCFastTransform.h"
#include "GPUO2InterfaceRefit.h"
#include "GlobalTracking/MatchTPCITSParams.h"
#include "CommonDataFormat/FlatHisto2D.h"

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

constexpr int Zero = 0;
constexpr int MinusOne = -1;
constexpr int MinusTen = -10;
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
  o2::math_utils::Bracketf_t timeBins; ///< bracketing time-bins
  int sourceID = 0;                    ///< track origin id
  float zMin = 0;                      // min possible Z of this track
  float zMax = 0;                      // max possible Z of this track
  int matchID = MinusOne;              ///< entry (non if MinusOne) of its matchTPC struct in the mMatchesTPC
  TrackLocTPC(const o2::track::TrackParCov& src, int tid) : o2::track::TrackParCov(src), sourceID(tid) {}
  TrackLocTPC() = default;
  ClassDefNV(TrackLocTPC, 1);
};

///< ITS track outward parameters propagated to reference X, with time bracket and index of
///< original track in the currently loaded ITS reco output
struct TrackLocITS : public o2::track::TrackParCov {
  int sourceID = 0;       ///< track origin id
  int roFrame = MinusOne; ///< ITS readout frame assigned to this track
  int matchID = MinusOne; ///< entry (non if MinusOne) of its matchCand struct in the mMatchesITS
  TrackLocITS(const o2::track::TrackParCov& src, int tid) : o2::track::TrackParCov(src), sourceID(tid) {}
  TrackLocITS() = default;
  ClassDefNV(TrackLocITS, 1);
};

///< record TPC or ITS track associated with single ITS or TPC track and reference on
///< the next (worse chi2) MatchRecord of the same TPC or ITS track
struct MatchRecord {
  float chi2 = -1.f;        ///< matching chi2
  int partnerID = MinusOne; ///< id of parnter track entry in mTPCWork or mITSWork containers
  int nextRecID = MinusOne; ///< index of eventual next record
  int matchedIC = MinusOne; ///< index of eventually matched InteractionCandidate
  MatchRecord(int partID, float chi2match, int nxt = MinusOne, int candIC = MinusOne) : partnerID(partID), chi2(chi2match), nextRecID(nxt), matchedIC(candIC) {}
  MatchRecord() = default;

  bool isBetter(float otherChi2, int otherIC = MinusOne) const
  { // prefer record with matched IC candidate, otherwise, better chi2
    if (otherIC == MinusOne) {
      return matchedIC == MinusOne ? chi2 < otherChi2 : true;
    } else {
      return matchedIC == MinusOne ? false : chi2 < otherChi2;
    }
  }

  bool isBetter(const MatchRecord& other) const
  {
    return isBetter(other.chi2, other.matchedIC);
  }
};

///< Link of the AfterBurner track: update at sertain cluster
///< original track in the currently loaded TPC reco output
struct ABTrackLink : public o2::track::TrackParCov {
  static constexpr int Disabled = -2;
  int clID = MinusOne;     ///< ID of the attached cluster, MinusTen is for dummy layer above Nlr, MinusOne: no attachment on this layer
  int parentID = MinusOne; ///< ID of the parent link (on prev layer) or parent TPC seed
  int nextOnLr = MinusOne; ///< ID of the next (in quality) link on the same layer
  int icCandID = MinusOne; ///< ID of the interaction candidate this track belongs to
  uint8_t nDaughters = 0;  ///< number of daughter links on lower layers
  int8_t layerID = -1;     ///< layer ID
  uint8_t ladderID = 0xff; ///< ladder ID in the layer (used for seeds with 2 hits in the layer)
  float chi2 = 0.f;        ///< chi2 after update
#ifdef _ALLOW_DEBUG_AB_
  o2::track::TrackParCov seed; // seed before update
#endif
  ABTrackLink() = default;
  ABTrackLink(const o2::track::TrackParCov& src, int ic, int lr, int parid = MinusOne, int clid = MinusOne, float ch2 = 0.f)
    : o2::track::TrackParCov(src), clID(clid), parentID(parid), icCandID(ic), layerID(lr), chi2(ch2) {}
  bool isDisabled() const { return clID == Disabled; }
  void disable() { clID = Disabled; }
  bool isDummyTop() const { return clID == MinusTen; }
  float chi2Norm() const { return layerID < o2::its::RecoGeomHelper::getNLayers() ? chi2 / (o2::its::RecoGeomHelper::getNLayers() - layerID) : 999.; }
  float chi2NormPredict(float chi2cl) const { return (chi2 + chi2cl) / (1 + o2::its::RecoGeomHelper::getNLayers() - layerID); }
};

struct ABTrackLinksList {
  int trackID = MinusOne;                                     ///< TPC work track id
  int firstLinkID = MinusOne;                                 ///< 1st link added (used for fast clean-up)
  int bestOrdLinkID = MinusOne;                               ///< start of sorted list of ABOrderLink for final validation
  int8_t lowestLayer = o2::its::RecoGeomHelper::getNLayers(); // lowest layer reached
  int8_t status = MinusOne;                                   ///< status (RS TODO)
  std::array<int, o2::its::RecoGeomHelper::getNLayers() + 1> firstInLr;
  ABTrackLinksList(int id = MinusOne) : trackID(id)
  {
    firstInLr.fill(MinusOne);
  }
  bool isDisabled() const { return status == MinusTen; } // RS is this strict enough
  void disable() { status = MinusTen; }
  bool isValidated() const { return status == Validated; }
  void validate() { status = Validated; }
};

struct ABOrderLink {          ///< link used for cross-layer sorting of best ABTrackLinks of the ABTrackLinksList
  int trackLinkID = MinusOne; ///< ABTrackLink ID
  int nextLinkID = MinusOne;  ///< indext on the next ABOrderLink
  ABOrderLink() = default;
  ABOrderLink(int id, int nxt = MinusOne) : trackLinkID(id), nextLinkID(nxt) {}
};

//---------------------------------------------------
struct ABDebugLink : o2::BaseCluster<float> {
#ifdef _ALLOW_DEBUG_AB_
  // AB link debug version, kinematics BEFORE update is stored
  o2::track::TrackParCov seed;
#endif
  o2::MCCompLabel clLabel;
  float chi2 = 0.f;
  uint8_t lr = 0;

  ClassDefNV(ABDebugLink, 1);
};

struct ABDebugTrack {
  int trackID = 0;
  int icCand = 0;
  short order = 0;
  short valid = 0;
  o2::track::TrackParCov tpcSeed;
  o2::MCCompLabel tpcLabel;
  o2::math_utils::Bracket<float> icTimeBin;
  std::vector<ABDebugLink> links;
  float chi2 = 0;
  uint8_t nClusTPC = 0;
  uint8_t nClusITS = 0;
  uint8_t nClusITSCorr = 0;
  uint8_t sideAC = 0;

  ClassDefNV(ABDebugTrack, 1);
};

///< Link of the cluster used by AfterBurner: every used cluster will have 1 link for every update it did on some
///< AB seed. The link (index) points not on seed state it is updating but on the end-point of the seed (lowest layer reached)
struct ABClusterLink {
  static constexpr int Disabled = -2;
  int linkedABTrack = MinusOne;     ///< ID of final AB track hypothesis it updates
  int linkedABTrackList = MinusOne; ///< ID of the AB tracks list to which linkedABTrack belongs
  int nextABClusterLink = MinusOne; ///< ID of the next link of this cluster
  ABClusterLink() = default;
  ABClusterLink(int idLink, int idList) : linkedABTrack(idLink), linkedABTrackList(idList) {}
  bool isDisabled() const { return linkedABTrack == Disabled; }
  void disable() { linkedABTrack = Disabled; }
};

struct InteractionCandidate : public o2::InteractionRecord {
  o2::math_utils::Bracket<float> timeBins; // interaction time (int TPC time bins)
  int rofITS;                              // corresponding ITS ROF entry (in the ROFRecord vectors)
  uint32_t flag;                           // origin, etc.
  void* clRefPtr = nullptr;                // pointer on cluster references container (if any)
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
 public:
  using ITSCluster = o2::BaseCluster<float>;
  using ClusRange = o2::dataformats::RangeReference<int, int>;
  using MCLabContCl = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  using MCLabContTr = std::vector<o2::MCCompLabel>;
  using MCLabSpan = gsl::span<const o2::MCCompLabel>;
  using TPCTransform = o2::gpu::TPCFastTransform;
  using BracketF = o2::math_utils::Bracket<float>;
  using BracketIR = o2::math_utils::Bracket<o2::InteractionRecord>;
  using Params = o2::globaltracking::MatchTPCITSParams;
  using MatCorrType = o2::base::Propagator::MatCorrType;

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
  void buildABCluster2TracksLinks();
  float correctTPCTrack(o2::track::TrackParCov& trc, const TrackLocTPC& tTPC, const InteractionCandidate& cand) const;
  int checkABSeedFromLr(int lrSeed, int seedID, ABTrackLinksList& llist);
  void accountForOverlapsAB(int lrSeed);
  void mergeABSeedsOnOverlaps(int lr, ABTrackLinksList& llist);
  ABTrackLinksList& createABTrackLinksList(int tpcWID);
  ABTrackLinksList& getABTrackLinksList(int tpcWID) { return mABTrackLinksList[mTPCWork[tpcWID].matchID]; }
  void disableABTrackLinksList(int tpcWID);
  int registerABTrackLink(ABTrackLinksList& llist, const o2::track::TrackParCov& src, int ic, int lr, int parentID = -1, int clID = -1, float chi2Cl = 0.f);
  void printABTracksTree(const ABTrackLinksList& llist) const;
  void printABClusterUsage() const;
  void selectBestMatchesAB();
  bool validateABMatch(int ilink);
  void buildBestLinksList(int ilink);
  bool isBetter(float chi2A, float chi2B) { return chi2A < chi2B; } // RS TODO
  void dumpABTracksDebugTree(const ABTrackLinksList& llist);
  int prepareInteractionTimes();
  void destroyLastABTrackLinksList();
  void refitABTrack(int ibest) const;

  void setCosmics(bool v) { mCosmics = v; }
  bool isCosmics() const { return mCosmics; }

  ///< perform all initializations
  void init();

  ///< clear results of previous event reco
  void clear();

  ///< set InteractionRecods for the beginning of the TF
  void setStartIR(const o2::InteractionRecord& ir) { mStartIR = ir; }

  ///< set Bunch filling and init helpers for validation by BCs
  void setBunchFilling(const o2::BunchFilling& bf);

  ///< ITS readout mode
  void setITSTriggered(bool v) { mITSTriggered = v; }
  bool isITSTriggered() const { return mITSTriggered; }

  void setUseFT0(bool v) { mUseFT0 = v; }
  bool getUseFT0() const { return mUseFT0; }

  ///< set ITS ROFrame duration in microseconds
  void setITSROFrameLengthMUS(float fums) { mITSROFrameLengthMUS = fums; }
  ///< set ITS ROFrame duration in BC (continuous mode only)
  void setITSROFrameLengthInBC(int nbc);

  // ==================== >> DPL-driven input >> =======================

  ///< set flag to use MC truth
  void setMCTruthOn(bool v)
  {
    mMCTruthON = v;
  }

  ///< request VDrift calibration
  void setVDriftCalib(bool v)
  {
    mVDriftCalibOn = v;
  }

  ///< get histo for tgl differences for VDrift calibration
  auto getHistoDTgl() { return mHistoDTgl.get(); }

  ///< set input ITS tracks
  void setITSTracksInp(const gsl::span<const o2::its::TrackITS> inp)
  {
    mITSTracksArray = inp;
  }

  ///< set input ITS tracks cluster indices
  void setITSTrackClusIdxInp(const gsl::span<const int> inp)
  {
    mITSTrackClusIdx = inp;
  }

  ///< set input ITS tracks ROF records
  void setITSTrackROFRecInp(const gsl::span<const o2::itsmft::ROFRecord> inp)
  {
    mITSTrackROFRec = inp;
  }

  ///< set input ITS clusters
  void setITSClustersInp(const gsl::span<const ITSCluster> inp)
  {
    mITSClustersArray = inp;
  }

  ///< set input ITS clusters ROF records
  void setITSClusterROFRecInp(const gsl::span<const o2::itsmft::ROFRecord> inp)
  {
    mITSClusterROFRec = inp;
  }

  ///< set input TPC tracks
  void setTPCTracksInp(const gsl::span<const o2::tpc::TrackTPC> inp)
  {
    mTPCTracksArray = inp;
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

  ///< set input ITS track MC labels
  void setITSTrkLabelsInp(const MCLabSpan& lbl)
  {
    mITSTrkLabels = lbl;
  }

  ///< set input ITS clusters MC labels
  void setITSClsLabelsInp(const MCLabContCl* lbl)
  {
    mITSClsLabels = lbl;
  }

  ///< set input TPC track MC labels
  void setTPCTrkLabelsInp(const MCLabSpan& lbl)
  {
    mTPCTrkLabels = lbl;
  }

  ///< set input FIT info
  void setFITInfoInp(const gsl::span<const o2::ft0::RecPoints> inp)
  {
    mFITInfo = inp;
  }

  // ===================== << DPL-driven input << ========================

  ///< print settings
  void print() const;
  void printCandidatesTPC() const;
  void printCandidatesITS() const;

  std::vector<o2::dataformats::TrackTPCITS>& getMatchedTracks() { return mMatchedTracks; }
  MCLabContTr& getMatchedITSLabels() { return mOutITSLabels; }
  MCLabContTr& getMatchedTPCLabels() { return mOutTPCLabels; }

  //>>> ====================== options =============================>>>
  void setUseMatCorrFlag(MatCorrType f) { mUseMatCorrFlag = f; }
  auto getUseMatCorrFlag() const { return mUseMatCorrFlag; }

  //<<< ====================== options =============================<<<

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
  void updateTimeDependentParams();

  int findLaddersToCheckBOn(int ilr, int lad0, const o2::math_utils::CircleXYf_t& circle, float errYFrac,
                            std::array<int, MaxLadderCand>& lad2Check) const;
  int findLaddersToCheckBOff(int ilr, int lad0, const o2::math_utils::IntervalXYf_t& trcLinPar, float errYFrac,
                             std::array<int, MatchTPCITS::MaxLadderCand>& lad2Check) const;

  int prepareTPCTracksAfterBurner();
  bool prepareTPCTracks();
  bool prepareITSTracks();
  bool prepareFITInfo();

  int preselectChipClusters(std::vector<int>& clVecOut, const ClusRange& clRange, const ITSChipClustersRefs& clRefs,
                            float trackY, float trackZ, float tolerY, float tolerZ, const o2::MCCompLabel& lblTrc) const;
  void fillClustersForAfterBurner(ITSChipClustersRefs& refCont, int rofStart, int nROFs = 1);
  void cleanAfterBurnerClusRefCache(int currentIC, int& startIC);
  void flagUsedITSClusters(const o2::its::TrackITS& track, int rofOffset);

  void doMatching(int sec);

  void refitWinners();
  bool refitTrackTPCITS(int iTPC, int& iITS);
  bool refitTPCInward(o2::track::TrackParCov& trcIn, float& chi2, float xTgt, int trcID, float timeTB) const;

  void selectBestMatches();
  bool validateTPCMatch(int iTPC);
  void removeITSfromTPC(int itsID, int tpcID);
  void removeTPCfromITS(int tpcID, int itsID);
  bool isValidatedTPC(const TrackLocTPC& t) const;
  bool isValidatedITS(const TrackLocITS& t) const;
  bool isDisabledTPC(const TrackLocTPC& t) const;
  bool isDisabledITS(const TrackLocITS& t) const;

  int compareTPCITSTracks(const TrackLocITS& tITS, const TrackLocTPC& tTPC, float& chi2) const;
  float getPredictedChi2NoZ(const o2::track::TrackParCov& trITS, const o2::track::TrackParCov& trTPC) const;
  bool propagateToRefX(o2::track::TrackParCov& trc);
  void addLastTrackCloneForNeighbourSector(int sector);

  ///------------------- manipulations with matches records ----------------------
  bool registerMatchRecordTPC(int iITS, int iTPC, float chi2, int candIC = MinusOne);
  void registerMatchRecordITS(int iITS, int iTPC, float chi2, int candIC = MinusOne);
  void suppressMatchRecordITS(int iITS, int iTPC);

  ///< get number of matching records for TPC track
  int getNMatchRecordsTPC(const TrackLocTPC& tTPC) const;

  ///< get number of matching records for ITS track
  int getNMatchRecordsITS(const TrackLocITS& tITS) const;

  ///< convert TPC timebins bracket to IR bracket
  BracketIR tpcTimeBin2IRBracket(const BracketF tbrange);

  ///< convert TPC time bin to ITS ROFrame units
  int tpcTimeBin2ITSROFrame(float tbin) const
  {
    if (mITSTriggered) {
      return mITSROFofTPCBin[int(tbin > 0 ? tbin : 0)];
    }
    int rof = tbin > 0 ? tbin * mTPCBin2ITSROFrame : 0;
    // the rof is estimated continuous counter but the actual bins might have gaps (e.g. HB rejects etc)-> use mapping
    return rof < int(mITSTrackROFContMapping.size()) ? mITSTrackROFContMapping[rof] : mITSTrackROFContMapping.back();
  }

  ///< convert ITS ROFrame to TPC time bin units // TOREMOVE
  float itsROFrame2TPCTimeBin(int rof) const { return rof * mITSROFrame2TPCBin; }

  ///< convert time in microseconds to TPC time bin units
  float time2TPCTimeBin(float tms) const
  {
    return tms * mTPCTBinMUSInv;
  }

  ///< convert TPC time bin to microseconds
  float tpcTimeBin2MUS(float tbn) const
  {
    return tbn * mTPCTBinMUS;
  }

  ///< convert TPC time bin to nanoseconds
  float tpcTimeBin2NS(float tbn) const
  {
    return tbn * mTPCTBinNS;
  }

  ///< convert Interaction Record for TPC time bin units
  float intRecord2TPCTimeBin(const o2::InteractionRecord& bc) const
  {
    return time2TPCTimeBin(bc.differenceInBC(mStartIR) * o2::constants::lhc::LHCBunchSpacingNS * 1e-3);
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

  ///< correct TPC time0 (int TPC time-bins)
  float getTPCTrackCorrectedTimeBin(const o2::tpc::TrackTPC& trc, float delta) const
  {
    float timeTB = trc.getTime0();
    if (trc.hasASideClustersOnly()) {
      timeTB += delta;
    } else if (trc.hasCSideClustersOnly()) {
      timeTB -= delta;
    } else {
      // TODO : special treatment of tracks crossing the CE
    }
    return timeTB;
  }

  //================================================================

  bool mInitDone = false;  ///< flag init already done
  bool mFieldON = true;    ///< flag for field ON/OFF
  bool mCosmics = false;   ///< flag cosmics mode
  bool mMCTruthON = false; ///< flag availability of MC truth
  float mBz = 0;           ///< nominal Bz

  o2::InteractionRecord mStartIR{0, 0}; ///< IR corresponding to the start of the TF

  ///========== Parameters to be set externally, e.g. from CCDB ====================
  const Params* mParams = nullptr;
  const o2::ft0::InteractionTag* mFT0Params = nullptr;

  MatCorrType mUseMatCorrFlag = MatCorrType::USEMatCorrTGeo;

  bool mITSTriggered = false; ///< ITS readout is triggered
  bool mUseFT0 = false;       ///< FT0 information is available

  ///< do we use track Z difference to reject fake matches? makes sense for triggered mode only
  bool mCompareTracksDZ = false;

  float mSectEdgeMargin2 = 0.; ///< crude check if ITS track should be matched also in neighbouring sector

  ///< safety margin in TPC time bins when estimating TPC track tMin and tMax from
  ///< assigned time0 and its track Z position (converted from mTPCTimeEdgeZSafeMargin)
  float mTPCTimeEdgeTSafeMargin = 0.f;

  int mITSROFrameLengthInBC = 0;    ///< ITS RO frame in BC (for ITS cont. mode only)
  float mITSROFrameLengthMUS = -1.; ///< ITS RO frame in \mus
  float mTPCVDrift0 = -1.;          ///< TPC nominal drift speed in cm/microseconds
  float mTPCVDrift0Inv = -1.;       ///< inverse TPC nominal drift speed in cm/microseconds
  float mTPCTBinMUS = 0.;           ///< TPC time bin duration in microseconds
  float mTPCTBinNS = 0.;            ///< TPC time bin duration in ns
  float mTPCTBinMUSInv = 0.;        ///< inverse TPC time bin duration in microseconds
  float mITSROFrame2TPCBin = 0.;    ///< conversion coeff from ITS ROFrame units to TPC time-bin
  float mTPCBin2ITSROFrame = 0.;    ///< conversion coeff from TPC time-bin to ITS ROFrame units
  float mZ2TPCBin = 0.;             ///< conversion coeff from Z to TPC time-bin
  float mTPCBin2Z = 0.;             ///< conversion coeff from TPC time-bin to Z
  float mNTPCBinsFullDrift = 0.;    ///< max time bin for full drift
  float mTPCZMax = 0.;              ///< max drift length
  float mTPCmeanX0Inv = 1. / 31850.; ///< TPC gas 1/X0

  float mMinTPCTrackPtInv = 999.; ///< cutoff on TPC track inverse pT
  float mMinITSTrackPtInv = 999.; ///< cutoff on ITS track inverse pT

  bool mVDriftCalibOn = false;                                ///< flag to produce VDrift calibration data
  std::unique_ptr<o2::dataformats::FlatHisto2D_f> mHistoDTgl; ///< histo for VDrift calibration data

  std::unique_ptr<TPCTransform> mTPCTransform;         ///< TPC cluster transformation
  std::unique_ptr<o2::gpu::GPUParam> mTPCClusterParam; ///< TPC clusters error param
  std::unique_ptr<o2::gpu::GPUO2InterfaceRefit> mTPCRefitter; ///< TPC refitter used for TPC tracks refit during the reconstruction

  o2::BunchFilling mBunchFilling;
  std::array<int16_t, o2::constants::lhc::LHCMaxBunches> mClosestBunchAbove; // closest filled bunch from above
  std::array<int16_t, o2::constants::lhc::LHCMaxBunches> mClosestBunchBelow; // closest filled bunch from below

  ///>>>------ these are input arrays which should not be modified by the matching code
  //           since this info is provided by external device
  gsl::span<const o2::tpc::TrackTPC> mTPCTracksArray;       ///< input TPC tracks span
  gsl::span<const o2::tpc::TPCClRefElem> mTPCTrackClusIdx;  ///< input TPC track cluster indices span
  gsl::span<const o2::itsmft::ROFRecord> mITSTrackROFRec;   ///< input ITS tracks ROFRecord span
  gsl::span<const o2::its::TrackITS> mITSTracksArray;       ///< input ITS tracks span
  gsl::span<const int> mITSTrackClusIdx;                    ///< input ITS track cluster indices span
  gsl::span<const ITSCluster> mITSClustersArray;            ///< input ITS clusters span
  gsl::span<const o2::itsmft::ROFRecord> mITSClusterROFRec; ///< input ITS clusters ROFRecord span
  gsl::span<const o2::ft0::RecPoints> mFITInfo;             ///< optional input FIT info span

  gsl::span<const unsigned char> mTPCRefitterShMap; ///< externally set TPC clusters sharing map

  const o2::tpc::ClusterNativeAccess* mTPCClusterIdxStruct = nullptr; ///< struct holding the TPC cluster indices

  const MCLabContCl* mITSClsLabels = nullptr; ///< input ITS Cluster MC labels
  MCLabSpan mITSTrkLabels;                    ///< input ITS Track MC labels
  MCLabSpan mTPCTrkLabels;                    ///< input TPC Track MC labels
  /// <<<-----

  std::vector<InteractionCandidate> mInteractions;                     ///< possible interaction times
  std::vector<o2::dataformats::RangeRefComp<8>> mITSROFIntCandEntries; ///< entries of InteractionCandidate vector for every ITS ROF bin

  ///< container for record the match of TPC track to single ITS track
  std::vector<MatchRecord> mMatchRecordsTPC;
  ///< container for reference to MatchRecord involving particular ITS track
  std::vector<MatchRecord> mMatchRecordsITS;

  std::vector<int> mITSROFofTPCBin;    ///< aux structure for mapping of TPC time-bins on ITS ROFs
  std::vector<BracketF> mITSROFTimes;  ///< min/max times of ITS ROFs in TPC time-bins
  std::vector<TrackLocTPC> mTPCWork;   ///< TPC track params prepared for matching
  std::vector<TrackLocITS> mITSWork;   ///< ITS track params prepared for matching
  MCLabContTr mTPCLblWork;             ///< TPC track labels
  MCLabContTr mITSLblWork;             ///< ITS track labels
  std::vector<float> mWinnerChi2Refit; ///< vector of refitChi2 for winners

  std::deque<ITSChipClustersRefs> mITSChipClustersRefs; ///< range of clusters for each chip in ITS (for AfterBurner)

  std::vector<ABTrackLinksList> mABTrackLinksList; ///< pool of ABTrackLinksList objects for every TPC track matched by AB
  std::vector<ABTrackLink> mABTrackLinks;          ///< pool AB track links
  std::vector<ABClusterLink> mABClusterLinks;      ///< pool AB cluster links
  std::vector<ABOrderLink> mABBestLinks;           ///< pool of ABOrder links for best links of the ABTrackLinksList
  std::vector<int> mABClusterLinkIndex;            ///< index of 1st ABClusterLink for every cluster used by AfterBurner, -1: unused, -10: used by external ITS tracks
  int mMaxABLinksOnLayer = 20;                     ///< max number of candidate links per layer
  int mMaxABFinalHyp = 10;                         ///< max number of final hypotheses to consider

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

  /// mapping for tracks' continuos ROF cycle to actual continuous readout ROFs with eventual gaps
  std::vector<int> mITSTrackROFContMapping;

  ///< outputs tracks container
  std::vector<o2::dataformats::TrackTPCITS> mMatchedTracks;
  MCLabContTr mOutITSLabels; ///< ITS label of matched track
  MCLabContTr mOutTPCLabels; ///< TPC label of matched track

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

  enum TimerIDs { SWTot,
                  SWPrepITS,
                  SWPrepTPC,
                  SWDoMatching,
                  SWSelectBest,
                  SWRefit,
                  SWIO,
                  SWDBG,
                  NStopWatches };
  static constexpr std::string_view TimerName[] = {"Total", "PrepareITS", "PrepareTPC", "DoMatching", "SelectBest", "Refit", "IO", "Debug"};
  TStopwatch mTimer[NStopWatches];

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
  int topID = MinusOne, next = tITS.matchID; // ITS MatchRecord
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

//______________________________________________
inline void MatchTPCITS::flagUsedITSClusters(const o2::its::TrackITS& track, int rofOffset)
{
  // flag clusters used by this track
  int clEntry = track.getFirstClusterEntry();
  for (int icl = track.getNumberOfClusters(); icl--;) {
    mABClusterLinkIndex[rofOffset + mITSTrackClusIdx[clEntry++]] = MinusTen;
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
    const auto& cls = mITSClustersArray[clID];
    float dz = trackZ - cls.getZ();
    auto label = mITSClsLabels->getLabels(clID)[0]; // tmp
    //    if (!(label == lblTrc)) {
    //      continue; // tmp
    //    }
    LOG(DEBUG) << "cl" << icl << '/' << clID << " " << label
               << " dZ: " << dz << " [" << tolerZ << "| dY: " << trackY - cls.getY() << " [" << tolerY << "]";
    if (dz > tolerZ) {
      float clsZ = cls.getZ();
      LOG(DEBUG) << "Skip the rest since " << trackZ << " > " << clsZ << "\n";
      break;
    } else if (dz < -tolerZ) {
      LOG(DEBUG) << "Skip cluster dz=" << dz << " Ztr=" << trackZ << " zCl=" << cls.getZ();
      continue;
    }
    if (fabs(trackY - cls.getY()) > tolerY) {
      LOG(DEBUG) << "Skip cluster dy= " << trackY - cls.getY() << " Ytr=" << trackY << " yCl=" << cls.getY();
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
  while (startIC < currentIC && mInteractions[currentIC].timeBins.getMin() - mInteractions[startIC].timeBins.getMax() > MinTBToCleanCache) {
    LOG(INFO) << "CAN REMOVE CACHE FOR " << startIC << " curent IC=" << currentIC;
    while (mInteractions[startIC].clRefPtr == &mITSChipClustersRefs.front()) {
      LOG(INFO) << "Reset cache pointer" << mInteractions[startIC].clRefPtr << " for IC=" << startIC;
      mInteractions[startIC++].clRefPtr = nullptr;
    }
    LOG(INFO) << "Reset cache slot " << &mITSChipClustersRefs.front();
    mITSChipClustersRefs.pop_front();
  }
}

//______________________________________________
inline void MatchTPCITS::destroyLastABTrackLinksList()
{
  // Profit from the links of the last ABTrackLinksList having been added in the very end of mABTrackLinks
  // and eliminate them also removing the last ABTrackLinksList.
  // This method should not be called after buildABCluster2TracksLinks!!!
  const auto& llist = mABTrackLinksList.back();
  mABTrackLinks.resize(llist.firstLinkID);
  mABTrackLinksList.pop_back();
}

} // namespace globaltracking
} // namespace o2

#endif
