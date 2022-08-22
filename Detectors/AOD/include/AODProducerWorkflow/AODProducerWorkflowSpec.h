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

/// @file   AODProducerWorkflowSpec.h

#ifndef O2_AODPRODUCER_WORKFLOW_SPEC
#define O2_AODPRODUCER_WORKFLOW_SPEC

#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsFT0/RecPoints.h"
#include "DataFormatsFDD/RecPoint.h"
#include "DataFormatsFV0/RecPoints.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTRD/TrackTRD.h"
#include "DataFormatsZDC/BCRecData.h"
#include "DataFormatsEMCAL/EventHandler.h"
#include "DataFormatsPHOS/EventHandler.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisHelpers.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "Steer/MCKinematicsReader.h"
#include "TMap.h"
#include "TStopwatch.h"

#include <boost/functional/hash.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/unordered_map.hpp>
#include <string>
#include <vector>

using namespace o2::framework;
using GID = o2::dataformats::GlobalTrackID;
using GIndex = o2::dataformats::VtxTrackIndex;
using DataRequest = o2::globaltracking::DataRequest;

namespace o2::aodproducer
{

typedef boost::tuple<int, int, int> Triplet_t;

struct TripletHash {
  std::size_t operator()(Triplet_t const& e) const
  {
    std::size_t seed = 0;
    boost::hash_combine(seed, e.get<0>());
    boost::hash_combine(seed, e.get<1>());
    boost::hash_combine(seed, e.get<2>());
    return seed;
  }
};

struct TripletEqualTo {
  bool operator()(Triplet_t const& x, Triplet_t const& y) const
  {
    return (x.get<0>() == y.get<0>() &&
            x.get<1>() == y.get<1>() &&
            x.get<2>() == y.get<2>());
  }
};

typedef boost::unordered_map<Triplet_t, int, TripletHash, TripletEqualTo> TripletsMap_t;

class AODProducerWorkflowDPL : public Task
{
 public:
  AODProducerWorkflowDPL(GID::mask_t src, std::shared_ptr<DataRequest> dataRequest, std::shared_ptr<o2::base::GRPGeomRequest> gr, bool enableSV, std::string resFile, bool useMC = true) : mInputSources(src), mDataRequest(dataRequest), mGGCCDBRequest(gr), mEnableSV(enableSV), mResFile{resFile}, mUseMC(useMC) {}
  ~AODProducerWorkflowDPL() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;

 private:
  // takes a local vertex timing in NS and converts to a lobal BC information relative to start of timeframe
  uint64_t relativeTime_to_LocalBC(double relativeTimeStampInNS) const
  {
    return relativeTimeStampInNS > 0. ? std::round(relativeTimeStampInNS / o2::constants::lhc::LHCBunchSpacingNS) : 0;
  }
  // takes a local vertex timing in NS and converts to a global BC information
  uint64_t relativeTime_to_GlobalBC(double relativeTimeStampInNS) const
  {
    return std::uint64_t(mStartIR.toLong()) + relativeTime_to_LocalBC(relativeTimeStampInNS);
  }

  bool mUseMC = true;
  bool mEnableSV = true;             // enable secondary vertices
  const float cSpeed = 0.029979246f; // speed of light in TOF units

  GID::mask_t mInputSources;
  int64_t mTFNumber{-1};
  int mRunNumber{-1};
  int mTruncate{1};
  int mRecoOnly{0};
  o2::InteractionRecord mStartIR{}; // TF 1st IR
  TString mResFile{"AO2D"};
  TString mLPMProdTag{""};
  TString mAnchorPass{""};
  TString mAnchorProd{""};
  TString mRecoPass{""};
  TStopwatch mTimer;

  // unordered map connects global indices and table indices of barrel tracks
  std::unordered_map<GIndex, int> mGIDToTableID;
  int mTableTrID{0};
  // unordered map connects global indices and table indices of fwd tracks
  std::unordered_map<GIndex, int> mGIDToTableFwdID;
  int mTableTrFwdID{0};
  // unordered map connects global indices and table indices of MFT tracks
  std::unordered_map<GIndex, int> mGIDToTableMFTID;
  int mTableTrMFTID{0};
  // unordered map connects global indices and table indices of vertices
  std::unordered_map<GIndex, int> mVtxToTableCollID;
  int mTableCollID{0};
  // unordered map connects global indices and table indices of V0s (needed for cascades references)
  std::unordered_map<GIndex, int> mV0ToTableID;
  int mTableV0ID{0};

  //  std::unordered_map<int, int> mIndexTableFwd;
  std::vector<int> mIndexTableFwd;
  int mIndexFwdID{0};
  //  std::unordered_map<int, int> mIndexTableMFT;
  std::vector<int> mIndexTableMFT;
  int mIndexMFTID{0};

  // zdc helper maps to avoid a number of "if" statements
  // when filling ZDC table
  map<string, float> mZDCEnergyMap; // mapping detector name to a corresponding energy
  map<string, float> mZDCTDCMap;    // mapping TDC channel to a corresponding TDC value

  std::vector<uint16_t> mITSTPCTRDTriggers; // mapping from TRD tracks ID to corresponding trigger (for tracks time extraction)
  std::vector<uint16_t> mTPCTRDTriggers;    // mapping from TRD tracks ID to corresponding trigger (for tracks time extraction)
  std::vector<uint16_t> mITSROFs;           // mapping from ITS tracks ID to corresponding ROF (for SA ITS tracks time extraction)
  std::vector<uint16_t> mMFTROFs;           // mapping from MFT tracks ID to corresponding ROF (for SA MFT tracks time extraction)
  std::vector<uint16_t> mMCHROFs;           // mapping from MCH tracks ID to corresponding ROF (for SA MCH tracks time extraction)
  double mITSROFrameHalfLengthNS = -1;      // ITS ROF half length
  double mMFTROFrameHalfLengthNS = -1;      // ITS ROF half length
  double mNSigmaTimeTrack = -1;             // number track errors sigmas (for gaussian errors only) used in track-vertex matching
  double mTimeMarginTrackTime = -1;         // safety margin in NS used for track-vertex matching (additive to track uncertainty)
  double mTPCBinNS = -1;                    // inverse TPC time-bin in ns

  TripletsMap_t mToStore;

  // MC production metadata holder
  TMap mMetaData;

  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;

  static constexpr int TOFTimePrecPS = 16; // required max error in ps for TOF tracks
  // truncation is enabled by default
  uint32_t mCollisionPosition = 0xFFFFFFF0;    // 19 bits mantissa
  uint32_t mCollisionPositionCov = 0xFFFFE000; // 10 bits mantissa
  uint32_t mTrackX = 0xFFFFFFF0;               // 19 bits
  uint32_t mTrackAlpha = 0xFFFFFFF0;           // 19 bits
  uint32_t mTrackSnp = 0xFFFFFF00;             // 15 bits
  uint32_t mTrackTgl = 0xFFFFFF00;             // 15 bits
  uint32_t mTrack1Pt = 0xFFFFFC00;             // 13 bits
  uint32_t mTrackCovDiag = 0xFFFFFF00;         // 15 bits
  uint32_t mTrackChi2 = 0xFFFF0000;            // 7 bits
  uint32_t mTrackCovOffDiag = 0xFFFF0000;      // 7 bits
  uint32_t mTrackSignal = 0xFFFFFF00;          // 15 bits
  uint32_t mTrackTime = 0xFFFFFFFF;            // use full float precision for time
  uint32_t mTrackTimeError = 0xFFFFFF00;       // 15 bits
  uint32_t mTrackPosEMCAL = 0xFFFFFF00;        // 15 bits
  uint32_t mTracklets = 0xFFFFFF00;            // 15 bits
  uint32_t mMcParticleW = 0xFFFFFFF0;          // 19 bits
  uint32_t mMcParticlePos = 0xFFFFFFF0;        // 19 bits
  uint32_t mMcParticleMom = 0xFFFFFFF0;        // 19 bits
  uint32_t mCaloAmp = 0xFFFFFF00;              // 15 bits
  uint32_t mCaloTime = 0xFFFFFF00;             // 15 bits
  uint32_t mMuonTr1P = 0xFFFFFC00;             // 13 bits
  uint32_t mMuonTrThetaX = 0xFFFFFF00;         // 15 bits
  uint32_t mMuonTrThetaY = 0xFFFFFF00;         // 15 bits
  uint32_t mMuonTrZmu = 0xFFFFFFF0;            // 19 bits
  uint32_t mMuonTrBend = 0xFFFFFFF0;           // 19 bits
  uint32_t mMuonTrNonBend = 0xFFFFFFF0;        // 19 bits
  uint32_t mMuonTrCov = 0xFFFF0000;            // 7 bits
  uint32_t mMuonCl = 0xFFFFFF00;               // 15 bits
  uint32_t mMuonClErr = 0xFFFF0000;            // 7 bits
  uint32_t mV0Time = 0xFFFFF000;               // 11 bits
  uint32_t mFDDTime = 0xFFFFF000;              // 11 bits
  uint32_t mT0Time = 0xFFFFFF00;               // 15 bits
  uint32_t mV0Amplitude = 0xFFFFF000;          // 11 bits
  uint32_t mFDDAmplitude = 0xFFFFF000;         // 11 bits
  uint32_t mT0Amplitude = 0xFFFFF000;          // 11 bits
  int mCTPReadout = 0;                         // 0 = use CTP readout from CTP; 1 = create CTP readout
  // helper struct for extra info in fillTrackTablesPerCollision()
  struct TrackExtraInfo {
    float tpcInnerParam = 0.f;
    uint32_t flags = 0;
    uint8_t itsClusterMap = 0;
    uint8_t tpcNClsFindable = 0;
    int8_t tpcNClsFindableMinusFound = 0;
    int8_t tpcNClsFindableMinusCrossedRows = 0;
    uint8_t tpcNClsShared = 0;
    uint8_t trdPattern = 0;
    float itsChi2NCl = -999.f;
    float tpcChi2NCl = -999.f;
    float trdChi2 = -999.f;
    float tofChi2 = -999.f;
    float tpcSignal = -999.f;
    float trdSignal = -999.f;
    float length = -999.f;
    float tofExpMom = -999.f;
    float trackEtaEMCAL = -999.f;
    float trackPhiEMCAL = -999.f;
    float trackTime = -999.f;
    float trackTimeRes = -999.f;
    int bcSlice[2] = {-1, -1};
  };

  // helper struct for addToFwdTracksTable()
  struct FwdTrackInfo {
    uint8_t trackTypeId = 0;
    float x = 0.f;
    float y = 0.f;
    float z = 0.f;
    float rabs = 0.f;
    float phi = 0.f;
    float tanl = 0.f;
    float invqpt = 0.f;
    float chi2 = 0.f;
    float pdca = 0.f;
    int nClusters = -1;
    float chi2matchmchmid = -1.0;
    float chi2matchmchmft = -1.0;
    float matchscoremchmft = -1.0;
    int matchmfttrackid = -1;
    int matchmchtrackid = -1;
    uint16_t mchBitMap = 0;
    uint8_t midBitMap = 0;
    uint32_t midBoards = 0;
    float trackTime = -999.f;
    float trackTimeRes = -999.f;
  };

  // helper struct for addToFwdTracksTable()
  struct FwdTrackCovInfo {
    float sigX = 0.f;
    float sigY = 0.f;
    float sigPhi = 0.f;
    float sigTgl = 0.f;
    float sig1Pt = 0.f;
    int8_t rhoXY = 0;
    int8_t rhoPhiX = 0;
    int8_t rhoPhiY = 0;
    int8_t rhoTglX = 0;
    int8_t rhoTglY = 0;
    int8_t rhoTglPhi = 0;
    int8_t rho1PtX = 0;
    int8_t rho1PtY = 0;
    int8_t rho1PtPhi = 0;
    int8_t rho1PtTgl = 0;
  };

  // helper struct for mc track labels
  // using -1 as dummies for AOD
  struct MCLabels {
    uint32_t labelID = -1;
    uint32_t labelITS = -1;
    uint32_t labelTPC = -1;
    uint16_t labelMask = 0;
    uint8_t fwdLabelMask = 0;
  };

  void updateTimeDependentParams(ProcessingContext& pc);

  void addRefGlobalBCsForTOF(const o2::dataformats::VtxTrackRef& trackRef, const gsl::span<const GIndex>& GIndices,
                             const o2::globaltracking::RecoContainer& data, std::map<uint64_t, int>& bcsMap);
  void createCTPReadout(const o2::globaltracking::RecoContainer& recoData, std::vector<o2::ctp::CTPDigit>& ctpDigits, ProcessingContext& pc);
  void collectBCs(const o2::globaltracking::RecoContainer& data,
                  const std::vector<o2::InteractionTimeRecord>& mcRecords,
                  std::map<uint64_t, int>& bcsMap);

  uint64_t getTFNumber(const o2::InteractionRecord& tfStartIR, int runNumber);
  template <typename TracksCursorType, typename TracksCovCursorType>
  void addToTracksTable(TracksCursorType& tracksCursor, TracksCovCursorType& tracksCovCursor,
                        const o2::track::TrackParCov& track, int collisionID);

  template <typename TracksExtraCursorType>
  void addToTracksExtraTable(TracksExtraCursorType& tracksExtraCursor, TrackExtraInfo& extraInfoHolder);

  template <typename mftTracksCursorType, typename AmbigMFTTracksCursorType>
  void addToMFTTracksTable(mftTracksCursorType& mftTracksCursor, AmbigMFTTracksCursorType& ambigMFTTracksCursor,
                           GIndex trackID, const o2::globaltracking::RecoContainer& data, int collisionID,
                           std::uint64_t collisionBC, const std::map<uint64_t, int>& bcsMap);

  template <typename fwdTracksCursorType, typename fwdTracksCovCursorType, typename AmbigFwdTracksCursorType>
  void addToFwdTracksTable(fwdTracksCursorType& fwdTracksCursor, fwdTracksCovCursorType& fwdTracksCovCursor, AmbigFwdTracksCursorType& ambigFwdTracksCursor,
                           GIndex trackID, const o2::globaltracking::RecoContainer& data, int collisionID, std::uint64_t collisionBC, const std::map<uint64_t, int>& bcsMap);

  TrackExtraInfo processBarrelTrack(int collisionID, std::uint64_t collisionBC, GIndex trackIndex, const o2::globaltracking::RecoContainer& data, const std::map<uint64_t, int>& bcsMap);

  void cacheTriggers(const o2::globaltracking::RecoContainer& recoData);

  // helper for track tables
  // * fills tables collision by collision
  // * interaction time is for TOF information
  template <typename TracksCursorType, typename TracksCovCursorType, typename TracksExtraCursorType, typename AmbigTracksCursorType,
            typename MFTTracksCursorType, typename AmbigMFTTracksCursorType,
            typename FwdTracksCursorType, typename FwdTracksCovCursorType, typename AmbigFwdTracksCursorType>
  void fillTrackTablesPerCollision(int collisionID,
                                   std::uint64_t collisionBC,
                                   const o2::dataformats::VtxTrackRef& trackRef,
                                   const gsl::span<const GIndex>& GIndices,
                                   const o2::globaltracking::RecoContainer& data,
                                   TracksCursorType& tracksCursor,
                                   TracksCovCursorType& tracksCovCursor,
                                   TracksExtraCursorType& tracksExtraCursor,
                                   AmbigTracksCursorType& ambigTracksCursor,
                                   MFTTracksCursorType& mftTracksCursor,
                                   AmbigMFTTracksCursorType& ambigMFTTracksCursor,
                                   FwdTracksCursorType& fwdTracksCursor,
                                   FwdTracksCovCursorType& fwdTracksCovCursor,
                                   AmbigFwdTracksCursorType& ambigFwdTracksCursor,
                                   const std::map<uint64_t, int>& bcsMap);

  void fillIndexTablesPerCollision(const o2::dataformats::VtxTrackRef& trackRef, const gsl::span<const GIndex>& GIndices, const o2::globaltracking::RecoContainer& data);

  template <typename V0CursorType, typename CascadeCursorType>
  void fillSecondaryVertices(const o2::globaltracking::RecoContainer& data, V0CursorType& v0Cursor, CascadeCursorType& cascadeCursor);

  template <typename MCParticlesCursorType>
  void fillMCParticlesTable(o2::steer::MCKinematicsReader& mcReader,
                            const MCParticlesCursorType& mcParticlesCursor,
                            const gsl::span<const o2::dataformats::VtxTrackRef>& primVer2TRefs,
                            const gsl::span<const GIndex>& GIndices,
                            const o2::globaltracking::RecoContainer& data,
                            const std::map<std::pair<int, int>, int>& mcColToEvSrc);

  template <typename MCTrackLabelCursorType, typename MCMFTTrackLabelCursorType, typename MCFwdTrackLabelCursorType>
  void fillMCTrackLabelsTable(const MCTrackLabelCursorType& mcTrackLabelCursor,
                              const MCMFTTrackLabelCursorType& mcMFTTrackLabelCursor,
                              const MCFwdTrackLabelCursorType& mcFwdTrackLabelCursor,
                              const o2::dataformats::VtxTrackRef& trackRef,
                              const gsl::span<const GIndex>& primVerGIs,
                              const o2::globaltracking::RecoContainer& data);

  std::uint64_t fillBCSlice(int (&slice)[2], double tmin, double tmax, const std::map<uint64_t, int>& bcsMap) const;

  // helper for tpc clusters
  void countTPCClusters(const o2::tpc::TrackTPC& track,
                        const gsl::span<const o2::tpc::TPCClRefElem>& tpcClusRefs,
                        const gsl::span<const unsigned char>& tpcClusShMap,
                        const o2::tpc::ClusterNativeAccess& tpcClusAcc,
                        uint8_t& shared, uint8_t& found, uint8_t& crossed);

  // helper for trd pattern
  uint8_t getTRDPattern(const o2::trd::TrackTRD& track);

  template <typename TEventHandler, typename TCaloCells, typename TCaloTriggerRecord, typename TCaloCursor, typename TCaloTRGTableCursor>
  void fillCaloTable(TEventHandler* caloEventHandler, const TCaloCells& calocells, const TCaloTriggerRecord& caloCellTRGR,
                     const TCaloCursor& caloCellCursor, const TCaloTRGTableCursor& caloCellTRGTableCursor,
                     std::map<uint64_t, int>& bcsMap, int8_t caloType);
};

/// create a processor spec
framework::DataProcessorSpec getAODProducerWorkflowSpec(GID::mask_t src, bool enableSV, bool useMC, std::string resFile);

// helper interface for calo cells to "befriend" emcal and phos cells
class CellHelper
{
 public:
  static int8_t getTriggerBits(const o2::emcal::Cell& cell)
  {
    return 0; // dummy value
  }

  static int8_t getTriggerBits(const o2::phos::Cell& cell)
  {
    return (cell.getType() == o2::phos::ChannelType_t::TRU2x2) ? 0 : 1;
  }

  static int16_t getCellNumber(const o2::emcal::Cell& cell)
  {
    return cell.getTower();
  }

  static int16_t getCellNumber(const o2::phos::Cell& cell)
  {
    return cell.getAbsId();
  }
  // If this cell - trigger one?
  static bool isTRU(const o2::emcal::Cell& cell)
  {
    return cell.getTRU();
  }

  static bool isTRU(const o2::phos::Cell& cell)
  {
    return cell.getTRU();
  }

  static int16_t getFastOrAbsID(const o2::emcal::Cell& cell)
  {
    return 0; // dummy value
  }

  static int16_t getFastOrAbsID(const o2::phos::Cell& cell)
  {
    return cell.getTRUId();
  }

  static float getAmplitude(const o2::emcal::Cell& cell)
  {
    return cell.getAmplitude();
  }

  static float getAmplitude(const o2::phos::Cell& cell)
  {
    return cell.getEnergy();
  }

  static int16_t getLnAmplitude(const o2::emcal::Cell& cell)
  {
    return 0; // dummy value
  }

  static int16_t getLnAmplitude(const o2::phos::Cell& cell)
  {
    return cell.getEnergy(); // dummy value
  }

  static float getTimeStamp(const o2::emcal::Cell& cell)
  {
    return cell.getTimeStamp();
  }

  static float getTimeStamp(const o2::phos::Cell& cell)
  {
    return cell.getTime();
  }
};

} // namespace o2::aodproducer

#endif /* O2_AODPRODUCER_WORKFLOW_SPEC */
