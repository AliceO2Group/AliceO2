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
#include "CommonDataFormat/Pair.h"
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
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsITSMFT/TrkClusRef.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "TPCCalibration/CorrectionMapsHelper.h"

class TTree;

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
class VDriftCorrFact;
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
  enum Constraint_t : uint8_t { Constrained,
                                ASide,
                                CSide };
  o2::math_utils::Bracketf_t tBracket;  ///< bracketing time in \mus
  float time0 = 0.f;                    ///< nominal time in \mus since start of TF (time0 for bare TPC tracks, constrained time for TRD/TOF constrained ones)
  float timeErr = 0.f;                  ///< time sigma (makes sense for constrained tracks only)
  int sourceID = 0;                     ///< TPC track origin in
  o2::dataformats::GlobalTrackID gid{}; // global track source ID (TPC track may be part of it)
  int matchID = MinusOne;              ///< entry (non if MinusOne) of its matchTPC struct in the mMatchesTPC
  Constraint_t constraint{Constrained};

  float getCorrectedTime(float dt) const // return time0 corrected for extra drift (to match certain Z)
  {
    return constraint == Constrained ? time0 : (constraint == ASide ? time0 + dt : time0 - dt);
  }
  float getSignedDT(float dt) const // account for TPC side in time difference for dt=external_time - tpc.time0
  {
    return constraint == Constrained ? 0. : (constraint == ASide ? dt : -dt);
  }

  ClassDefNV(TrackLocTPC, 1);
};

///< ITS track outward parameters propagated to reference X, with time bracket and index of
///< original track in the currently loaded ITS reco output
struct TrackLocITS : public o2::track::TrackParCov {
  o2::math_utils::Bracketf_t tBracket; ///< bracketing time in \mus
  int sourceID = 0;       ///< track origin id
  int roFrame = MinusOne; ///< ITS readout frame assigned to this track
  int matchID = MinusOne; ///< entry (non if MinusOne) of its matchCand struct in the mMatchesITS

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
  int clID = MinusOne;     ///< ID of the attached cluster, MinusOne: no attachment on this layer
  int parentID = MinusOne; ///< ID of the parent link (on prev layer) or parent TPC seed
  int nextOnLr = MinusOne; ///< ID of the next (in quality) link on the same layer
  uint8_t nDaughters = 0;  ///< number of daughter links on lower layers
  int8_t layerID = -1;     ///< layer ID
  int8_t nContLayers = 0;  ///< number of contributing layers
  uint8_t ladderID = 0xff; ///< ladder ID in the layer (used for seeds with 2 hits in the layer)
  float chi2 = 0.f;        ///< chi2 after update

  ABTrackLink(const o2::track::TrackParCov& tr, int cl, int parID, int nextID, int lr, int nc, int ld, float _chi2)
    : o2::track::TrackParCov(tr), clID(cl), parentID(parID), nextOnLr(nextID), layerID(int8_t(lr)), nContLayers(int8_t(nc)), ladderID(uint8_t(ld)), chi2(_chi2) {}

  bool isDisabled() const { return clID == Disabled; }
  void disable() { clID = Disabled; }
  bool isDummyTop() const { return clID == MinusTen; }
  float chi2Norm() const { return layerID < o2::its::RecoGeomHelper::getNLayers() ? chi2 / (o2::its::RecoGeomHelper::getNLayers() - layerID) : 999.; }
  float chi2NormPredict(float chi2cl) const { return (chi2 + chi2cl) / (1 + o2::its::RecoGeomHelper::getNLayers() - layerID); }
};

// AB primary seed: TPC track propagated to outermost ITS layer under specific InteractionCandidate hypothesis
struct TPCABSeed {
  static constexpr int8_t NeedAlternative = -3;
  int tpcWID = MinusOne;                                            ///< TPC track ID
  int ICCanID = MinusOne;                                           ///< interaction candidate ID (they are sorted in increasing time)
  int winLinkID = MinusOne;                                         ///< ID of the validated link
  int8_t lowestLayer = o2::its::RecoGeomHelper::getNLayers();       ///< lowest layer reached
  int8_t status = MinusOne;                                         ///< status (RS TODO)
  o2::track::TrackParCov track{};                                   ///< Seed propagated to the outer layer under certain time constraint
  std::array<int, o2::its::RecoGeomHelper::getNLayers()> firstInLr; ///< entry of 1st (best) hypothesis on each layer
  std::vector<ABTrackLink> trackLinks{};                            ///< links
  TPCABSeed(int id, int ic, const o2::track::TrackParCov& trc) : tpcWID(id), ICCanID(ic), track(trc)
  {
    firstInLr.fill(MinusOne);
  }
  bool isDisabled() const { return status == MinusTen; }
  void disable() { status = MinusTen; }
  bool isValidated() const { return status == Validated; }
  void validate(int lID)
  {
    winLinkID = lID;
    status = Validated;
  }
  bool needAlteranative() const { return status == NeedAlternative; }
  void setNeedAlternative() { status = NeedAlternative; }
  ABTrackLink& getLink(int i) { return trackLinks[i]; }
  const ABTrackLink& getLink(int i) const { return trackLinks[i]; }
  auto getBestLinkID() const
  {
    return lowestLayer < o2::its::RecoGeomHelper::getNLayers() ? firstInLr[lowestLayer] : -1;
  }
  bool checkLinkHasUsedClusters(int linkID, const std::vector<int>& clStatus) const
  {
    // check if some clusters used by the link or its parents are forbidden (already used by validatet track)
    while (linkID > MinusOne) {
      const auto& link = trackLinks[linkID];
      if (link.clID > MinusOne && clStatus[link.clID] != MinusOne) {
        return true;
      }
      linkID = link.parentID;
    }
    return false;
  }
  void flagLinkUsedClusters(int linkID, std::vector<int>& clStatus) const
  {
    // check if some clusters used by the link or its parents are forbidden (already used by validated track)
    while (linkID > MinusOne) {
      const auto& link = trackLinks[linkID];
      if (link.clID > MinusOne) {
        clStatus[link.clID] = MinusTen;
      }
      linkID = link.parentID;
    }
  }
};

struct InteractionCandidate : public o2::InteractionRecord {
  o2::math_utils::Bracketf_t tBracket;                // interaction time
  int rofITS;                                         // corresponding ITS ROF entry (in the ROFRecord vectors)
  uint32_t flag;                                      // origin, etc.
  o2::dataformats::RangeReference<int, int> seedsRef; // references to AB seeds
  InteractionCandidate() = default;
  InteractionCandidate(const o2::InteractionRecord& ir, float t, float dt, int rof, uint32_t f = 0) : o2::InteractionRecord(ir), tBracket(t - dt, t + dt), rofITS(rof), flag(f) {}
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

  static constexpr float XMatchingRef = 70.0;                            ///< reference radius to propage tracks for matching
  static constexpr float YMaxAtXMatchingRef = XMatchingRef * 0.17632698; ///< max Y in the sector at reference X

  static constexpr int MaxUpDnLadders = 3;                     // max N ladders to check up and down from selected one
  static constexpr int MaxLadderCand = 2 * MaxUpDnLadders + 1; // max ladders to check for matching clusters
  static constexpr int MaxSeedsPerLayer = 50;                  // TODO
  static constexpr int NITSLayers = o2::its::RecoGeomHelper::getNLayers();
  ///< perform matching for provided input
  void run(const o2::globaltracking::RecoContainer& inp);

  void setSkipTPCOnly(bool v) { mSkipTPCOnly = v; }
  void setCosmics(bool v) { mCosmics = v; }
  bool isCosmics() const { return mCosmics; }
  void setNThreads(int n);

  ///< perform all initializations
  void init();
  void end();

  ///< clear results of previous event reco
  void clear();

  ///< set Bunch filling and init helpers for validation by BCs
  void setBunchFilling(const o2::BunchFilling& bf);

  ///< ITS readout mode
  void setITSTriggered(bool v) { mITSTriggered = v; }
  bool isITSTriggered() const { return mITSTriggered; }

  void setUseFT0(bool v) { mUseFT0 = v; }
  bool getUseFT0() const { return mUseFT0; }

  void setUseBCFilling(bool v) { mUseBCFilling = v; }
  bool getUseBCFilling() const { return mUseBCFilling; }

  ///< set ITS ROFrame duration in microseconds
  void setITSROFrameLengthMUS(float fums);
  ///< set ITS ROFrame duration in BC (continuous mode only)
  void setITSROFrameLengthInBC(int nbc);

  // ==================== >> DPL-driven input >> =======================
  void setITSDictionary(const o2::itsmft::TopologyDictionary* d) { mITSDict = d; }

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
  void setTPCVDrift(const o2::tpc::VDriftCorrFact& v);
  void setTPCCorrMaps(o2::tpc::CorrectionMapsHelper* maph);

  ///< print settings
  void print() const;
  void printCandidatesTPC() const;
  void printCandidatesITS() const;

  const std::vector<o2::dataformats::TrackTPCITS>& getMatchedTracks() const { return mMatchedTracks; }
  const MCLabContTr& getMatchLabels() const { return mOutLabels; }
  const MCLabContTr& getABTrackletLabels() const { return mABTrackletLabels; }
  const std::vector<int>& getABTrackletClusterIDs() const { return mABTrackletClusterIDs; }
  const std::vector<o2::itsmft::TrkClusRef>& getABTrackletRefs() const { return mABTrackletRefs; }
  const std::vector<o2::dataformats::Pair<float, float>>& getTglITSTPC() const { return mTglITSTPC; }

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
  void fillTPCITSmatchTree(int itsID, int tpcID, int rejFlag, float chi2 = -1., float tCorr = 0.);
  void dumpWinnerMatches();
#endif

 private:
  void updateTimeDependentParams();

  int findLaddersToCheckBOn(int ilr, int lad0, const o2::math_utils::CircleXYf_t& circle, float errYFrac,
                            std::array<int, MaxLadderCand>& lad2Check) const;
  int findLaddersToCheckBOff(int ilr, int lad0, const o2::math_utils::IntervalXYf_t& trcLinPar, float errYFrac,
                             std::array<int, MatchTPCITS::MaxLadderCand>& lad2Check) const;
  bool prepareTPCData();
  bool prepareITSData();
  bool prepareFITData();
  int prepareInteractionTimes();
  int prepareTPCTracksAfterBurner();
  void addTPCSeed(const o2::track::TrackParCov& _tr, float t0, float terr, o2::dataformats::GlobalTrackID srcGID, int tpcID);

  int preselectChipClusters(std::vector<int>& clVecOut, const ClusRange& clRange, const ITSChipClustersRefs& itsChipClRefs,
                            float trackY, float trackZ, float tolerY, float tolerZ) const;
  void fillClustersForAfterBurner(int rofStart, int nROFs, ITSChipClustersRefs& itsChipClRefs);
  void flagUsedITSClusters(const o2::its::TrackITS& track);

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

  ///< convert time bracket to IR bracket
  BracketIR tBracket2IRBracket(const BracketF tbrange);

  ///< convert time to ITS ROFrame units in case of continuous ITS readout
  int time2ITSROFrameCont(float t) const
  {
    int rof = t > 0 ? t * mITSROFrameLengthMUSInv : 0;
    // the rof is estimated continuous counter but the actual bins might have gaps (e.g. HB rejects etc)-> use mapping
    return rof < int(mITSTrackROFContMapping.size()) ? mITSTrackROFContMapping[rof] : mITSTrackROFContMapping.back();
  }

  ///< convert time to ITS ROFrame units in case of triggered ITS readout
  int time2ITSROFrameTrig(float t, int start) const
  {
    while (start < mITSROFTimes.size()) {
      if (mITSROFTimes[start].getMax() > t) {
        return start;
      }
      start++;
    }
    return --start;
  }

  ///< convert TPC time bin to microseconds
  float tpcTimeBin2Z(float tbn) const { return tbn * mTPCBin2Z; }

  ///< convert TPC time bin to microseconds
  float tpcTimeBin2MUS(float tbn) const { return tbn * mTPCTBinMUS; }

  ///< convert TPC time bin to nanoseconds
  float tpcTimeBin2NS(float tbn) const { return tbn * mTPCTBinNS; }

  ///< convert Z interval to TPC time-bins
  float z2TPCBin(float z) const { return z * mZ2TPCBin; }

  ///< convert TPC time-bins to Z interval
  float tpcBin2Z(float t) const { return t * mTPCBin2Z; }

  ///< rought check of 2 track params difference, return -1,0,1 if it is <,within or > than tolerance
  int roughCheckDif(float delta, float toler, int rejFlag) const
  {
    return delta > toler ? rejFlag : (delta < -toler ? -rejFlag : Accept);
  }

  // ========================= AFTERBURNER =========================
  void runAfterBurner();
  int prepareABSeeds();
  void processABSeed(int sid, const ITSChipClustersRefs& itsChipClRefs);
  int followABSeed(const o2::track::TrackParCov& seed, const ITSChipClustersRefs& itsChipClRefs, int seedID, int lrID, TPCABSeed& ABSeed);
  int registerABTrackLink(TPCABSeed& ABSeed, const o2::track::TrackParCov& trc, int clID, int parentID, int lr, int laddID, float chi2Cl);
  bool isBetter(float chi2A, float chi2B) { return chi2A < chi2B; } // RS FIMXE TODO
  void refitABWinners();
  bool refitABTrack(int iITSAB, const TPCABSeed& seed);
  void accountForOverlapsAB(int lrSeed);
  float correctTPCTrack(o2::track::TrackParCov& trc, const TrackLocTPC& tTPC, const InteractionCandidate& cand) const; // RS FIXME will be needed for refit
  //================================================================

  bool mInitDone = false;  ///< flag init already done
  bool mFieldON = true;    ///< flag for field ON/OFF
  bool mCosmics = false;   ///< flag cosmics mode
  bool mMCTruthON = false; ///< flag availability of MC truth
  float mBz = 0;           ///< nominal Bz
  int mTFCount = 0;        ///< internal TF counter for debugger
  int mNThreads = 1;       ///< number of OMP threads
  o2::InteractionRecord mStartIR{0, 0}; ///< IR corresponding to the start of the TF

  ///========== Parameters to be set externally, e.g. from CCDB ====================
  const Params* mParams = nullptr;
  const o2::ft0::InteractionTag* mFT0Params = nullptr;

  MatCorrType mUseMatCorrFlag = MatCorrType::USEMatCorrTGeo;
  bool mUseBCFilling = false; ///< use BC filling for candidates validation
  bool mSkipTPCOnly = false;  ///< for test only: don't use TPC only tracks, use only external ones
  bool mITSTriggered = false; ///< ITS readout is triggered
  bool mUseFT0 = false;       ///< FT0 information is available

  ///< do we use track Z difference to reject fake matches? makes sense for triggered mode only
  bool mCompareTracksDZ = false;

  float mSectEdgeMargin2 = 0.; ///< crude check if ITS track should be matched also in neighbouring sector

  ///< safety margin in TPC time bins when estimating TPC track tMin and tMax from
  ///< assigned time0 and its track Z position (converted from mTPCTimeEdgeZSafeMargin)
  float mTPCTimeEdgeTSafeMargin = 0.f;
  float mTPCExtConstrainedNSigmaInv = 0.f; // inverse for NSigmas for TPC time-interval from external constraint time sigma
  int mITSROFrameLengthInBC = 0;    ///< ITS RO frame in BC (for ITS cont. mode only)
  float mITSROFrameLengthMUS = -1.; ///< ITS RO frame in \mus
  float mITSTimeResMUS = -1.;       ///< nominal ITS time resolution derived from ROF
  float mITSROFrameLengthMUSInv = -1.; ///< ITS RO frame in \mus inverse
  float mTPCVDriftRef = -1.;           ///< TPC nominal drift speed in cm/microseconds
  float mTPCVDriftCorrFact = 1.;       ///< TPC nominal correction factort (wrt ref)
  float mTPCVDrift = -1.;              ///< TPC drift speed in cm/microseconds
  float mTPCVDriftInv = -1.;           ///< inverse TPC nominal drift speed in cm/microseconds
  float mTPCTBinMUS = 0.;           ///< TPC time bin duration in microseconds
  float mTPCTBinNS = 0.;            ///< TPC time bin duration in ns
  float mTPCTBinMUSInv = 0.;        ///< inverse TPC time bin duration in microseconds
  float mZ2TPCBin = 0.;             ///< conversion coeff from Z to TPC time-bin
  float mTPCBin2Z = 0.;             ///< conversion coeff from TPC time-bin to Z
  float mNTPCBinsFullDrift = 0.;    ///< max time bin for full drift
  float mTPCZMax = 0.;              ///< max drift length
  float mTPCmeanX0Inv = 1. / 31850.; ///< TPC gas 1/X0

  float mMinTPCTrackPtInv = 999.; ///< cutoff on TPC track inverse pT
  float mMinITSTrackPtInv = 999.; ///< cutoff on ITS track inverse pT

  bool mVDriftCalibOn = false;                                ///< flag to produce VDrift calibration data
  o2::tpc::CorrectionMapsHelper* mTPCCorrMapsHelper = nullptr;

  std::unique_ptr<o2::gpu::GPUO2InterfaceRefit> mTPCRefitter; ///< TPC refitter used for TPC tracks refit during the reconstruction

  o2::BunchFilling mBunchFilling;
  std::array<int16_t, o2::constants::lhc::LHCMaxBunches> mClosestBunchAbove; // closest filled bunch from above
  std::array<int16_t, o2::constants::lhc::LHCMaxBunches> mClosestBunchBelow; // closest filled bunch from below

  const o2::itsmft::ChipMappingITS ITSChMap{};

  const o2::globaltracking::RecoContainer* mRecoCont = nullptr;
  ///>>>------ these are input arrays which should not be modified by the matching code
  //           since this info is provided by external device
  gsl::span<const o2::tpc::TrackTPC> mTPCTracksArray;       ///< input TPC tracks span
  gsl::span<const o2::tpc::TPCClRefElem> mTPCTrackClusIdx;  ///< input TPC track cluster indices span
  gsl::span<const o2::itsmft::ROFRecord> mITSTrackROFRec;   ///< input ITS tracks ROFRecord span
  gsl::span<const o2::its::TrackITS> mITSTracksArray;       ///< input ITS tracks span
  gsl::span<const int> mITSTrackClusIdx;                    ///< input ITS track cluster indices span
  std::vector<ITSCluster> mITSClustersArray;                ///< ITS clusters created in loadInput
  gsl::span<const o2::itsmft::ROFRecord> mITSClusterROFRec; ///< input ITS clusters ROFRecord span
  gsl::span<const o2::ft0::RecPoints> mFITInfo;             ///< optional input FIT info span

  gsl::span<const unsigned char> mTPCRefitterShMap; ///< externally set TPC clusters sharing map

  const o2::itsmft::TopologyDictionary* mITSDict{nullptr}; // cluster patterns dictionary

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

  ////  std::vector<int> mITSROFofTPCBin;    ///< aux structure for mapping of TPC time-bins on ITS ROFs
  std::vector<BracketF> mITSROFTimes;  ///< min/max times of ITS ROFs in \mus
  std::vector<TrackLocTPC> mTPCWork;   ///< TPC track params prepared for matching
  std::vector<TrackLocITS> mITSWork;   ///< ITS track params prepared for matching
  MCLabContTr mTPCLblWork;             ///< TPC track labels
  MCLabContTr mITSLblWork;             ///< ITS track labels
  std::vector<float> mWinnerChi2Refit; ///< vector of refitChi2 for winners

  // ------------------------------
  std::vector<TPCABSeed> mTPCABSeeds; ///< pool of primary TPC seeds for AB
  ///< indices of selected track entries in mTPCWork (for tracks selected by AfterBurner)
  std::vector<int> mTPCABIndexCache;
  std::vector<int> mABWinnersIDs;
  std::vector<int> mABTrackletClusterIDs;              ///< IDs of ITS clusters for AfterBurner winners
  std::vector<o2::itsmft::TrkClusRef> mABTrackletRefs; ///< references on AfterBurner winners clusters
  std::vector<int> mABClusterLinkIndex;            ///< index of 1st ABClusterLink for every cluster used by AfterBurner, -1: unused, -10: used by external ITS tracks
  MCLabContTr mABTrackletLabels;
  // ------------------------------

  ///< per sector indices of TPC track entry in mTPCWork
  std::array<std::vector<int>, o2::constants::math::NSectors> mTPCSectIndexCache;
  ///< per sector indices of ITS track entry in mITSWork
  std::array<std::vector<int>, o2::constants::math::NSectors> mITSSectIndexCache;

  ///< indices of 1st TPC tracks with time above the ITS ROF time
  std::array<std::vector<int>, o2::constants::math::NSectors> mTPCTimeStart;
  ///< indices of 1st entries of ITS tracks starting at given ROframe
  std::array<std::vector<int>, o2::constants::math::NSectors> mITSTimeStart;

  /// mapping for tracks' continuos ROF cycle to actual continuous readout ROFs with eventual gaps
  std::vector<int> mITSTrackROFContMapping;

  ///< outputs tracks container
  std::vector<o2::dataformats::TrackTPCITS> mMatchedTracks;
  MCLabContTr mOutLabels; ///< Labels: = TPC labels with flag isFake set in case of fake matching

  ///< container for <tglITS, tglTPC> pairs for vdrift calibration
  std::vector<o2::dataformats::Pair<float, float>> mTglITSTPC;

  o2::its::RecoGeomHelper mRGHelper; ///< helper for cluster and geometry access

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
                  SWABSeeds,
                  SWABMatch,
                  SWABWinners,
                  SWABRefit,
                  SWIO,
                  SWDBG,
                  NStopWatches };
  static constexpr std::string_view TimerName[] = {"Total", "PrepareITS", "PrepareTPC", "DoMatching", "SelectBest", "Refit",
                                                   "ABSeeds", "ABMatching", "ABWinners", "ABRefit", "IO", "Debug"};
  TStopwatch mTimer[NStopWatches];

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


} // namespace globaltracking
} // namespace o2

#endif
