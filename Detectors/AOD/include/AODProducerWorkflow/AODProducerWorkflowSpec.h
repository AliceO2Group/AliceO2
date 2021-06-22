// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AODProducerWorkflowSpec.h

#ifndef O2_AODPRODUCER_WORKFLOW_SPEC
#define O2_AODPRODUCER_WORKFLOW_SPEC

#include "DataFormatsFT0/RecPoints.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisHelpers.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "TStopwatch.h"
#include "CCDB/BasicCCDBManager.h"
#include "Steer/MCKinematicsReader.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"

#include <string>
#include <vector>
#include <boost/tuple/tuple.hpp>
#include <boost/unordered_map.hpp>
#include <boost/functional/hash.hpp>

using namespace o2::framework;
using GID = o2::dataformats::GlobalTrackID;
using DataRequest = o2::globaltracking::DataRequest;

namespace o2::aodproducer
{

using TracksTable = o2::soa::Table<o2::aod::track::CollisionId,
                                   o2::aod::track::TrackType,
                                   o2::aod::track::X,
                                   o2::aod::track::Alpha,
                                   o2::aod::track::Y,
                                   o2::aod::track::Z,
                                   o2::aod::track::Snp,
                                   o2::aod::track::Tgl,
                                   o2::aod::track::Signed1Pt>;

using TracksCovTable = o2::soa::Table<o2::aod::track::SigmaY,
                                      o2::aod::track::SigmaZ,
                                      o2::aod::track::SigmaSnp,
                                      o2::aod::track::SigmaTgl,
                                      o2::aod::track::Sigma1Pt,
                                      o2::aod::track::RhoZY,
                                      o2::aod::track::RhoSnpY,
                                      o2::aod::track::RhoSnpZ,
                                      o2::aod::track::RhoTglY,
                                      o2::aod::track::RhoTglZ,
                                      o2::aod::track::RhoTglSnp,
                                      o2::aod::track::Rho1PtY,
                                      o2::aod::track::Rho1PtZ,
                                      o2::aod::track::Rho1PtSnp,
                                      o2::aod::track::Rho1PtTgl>;

using TracksExtraTable = o2::soa::Table<o2::aod::track::TPCInnerParam,
                                        o2::aod::track::Flags,
                                        o2::aod::track::ITSClusterMap,
                                        o2::aod::track::TPCNClsFindable,
                                        o2::aod::track::TPCNClsFindableMinusFound,
                                        o2::aod::track::TPCNClsFindableMinusCrossedRows,
                                        o2::aod::track::TPCNClsShared,
                                        o2::aod::track::TRDPattern,
                                        o2::aod::track::ITSChi2NCl,
                                        o2::aod::track::TPCChi2NCl,
                                        o2::aod::track::TRDChi2,
                                        o2::aod::track::TOFChi2,
                                        o2::aod::track::TPCSignal,
                                        o2::aod::track::TRDSignal,
                                        o2::aod::track::TOFSignal,
                                        o2::aod::track::Length,
                                        o2::aod::track::TOFExpMom,
                                        o2::aod::track::TrackEtaEMCAL,
                                        o2::aod::track::TrackPhiEMCAL>;

using MFTTracksTable = o2::soa::Table<o2::aod::fwdtrack::CollisionId,
                                      o2::aod::fwdtrack::X,
                                      o2::aod::fwdtrack::Y,
                                      o2::aod::fwdtrack::Z,
                                      o2::aod::fwdtrack::Phi,
                                      o2::aod::fwdtrack::Tgl,
                                      o2::aod::fwdtrack::Signed1Pt,
                                      o2::aod::fwdtrack::NClusters,
                                      o2::aod::fwdtrack::Chi2>;

using MCParticlesTable = o2::soa::Table<o2::aod::mcparticle::McCollisionId,
                                        o2::aod::mcparticle::PdgCode,
                                        o2::aod::mcparticle::StatusCode,
                                        o2::aod::mcparticle::Flags,
                                        o2::aod::mcparticle::Mother0,
                                        o2::aod::mcparticle::Mother1,
                                        o2::aod::mcparticle::Daughter0,
                                        o2::aod::mcparticle::Daughter1,
                                        o2::aod::mcparticle::Weight,
                                        o2::aod::mcparticle::Px,
                                        o2::aod::mcparticle::Py,
                                        o2::aod::mcparticle::Pz,
                                        o2::aod::mcparticle::E,
                                        o2::aod::mcparticle::Vx,
                                        o2::aod::mcparticle::Vy,
                                        o2::aod::mcparticle::Vz,
                                        o2::aod::mcparticle::Vt>;

typedef boost::tuple<int, int, int> Triplet_t;

struct TripletHash : std::unary_function<Triplet_t, std::size_t> {
  std::size_t operator()(Triplet_t const& e) const
  {
    std::size_t seed = 0;
    boost::hash_combine(seed, e.get<0>());
    boost::hash_combine(seed, e.get<1>());
    boost::hash_combine(seed, e.get<2>());
    return seed;
  }
};

struct TripletEqualTo : std::binary_function<Triplet_t, Triplet_t, bool> {
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
  AODProducerWorkflowDPL(std::shared_ptr<DataRequest> dataRequest, bool fillSVertices) : mDataRequest(dataRequest), mFillSVertices(fillSVertices) {}
  ~AODProducerWorkflowDPL() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;

 private:
  int mFillTracksITS{1};
  int mFillTracksMFT{1};
  int mFillTracksTPC{0};
  int mFillTracksITSTPC{1};
  int64_t mTFNumber{-1};
  int mTruncate{1};
  int mRecoOnly{0};
  bool mFillSVertices{false};
  TStopwatch mTimer;

  std::shared_ptr<DataRequest> mDataRequest;

  // truncation is enabled by default
  uint32_t mCollisionPosition = 0xFFFFFFF0;    // 19 bits mantissa
  uint32_t mCollisionPositionCov = 0xFFFFE000; // 10 bits mantissa
  uint32_t mTrackX = 0xFFFFFFF0;               // 19 bits
  uint32_t mTrackAlpha = 0xFFFFFFF0;           // 19 bits
  uint32_t mTrackSnp = 0xFFFFFF00;             // 15 bits
  uint32_t mTrackTgl = 0xFFFFFF00;             // 15 bits
  uint32_t mTrack1Pt = 0xFFFFFC00;             // 13 bits
  uint32_t mTrackCovDiag = 0xFFFFFF00;         // 15 bits
  uint32_t mTrackCovOffDiag = 0xFFFF0000;      // 7 bits
  uint32_t mTrackSignal = 0xFFFFFF00;          // 15 bits
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

  void collectBCs(gsl::span<const o2::ft0::RecPoints>& ft0RecPoints,
                  gsl::span<const o2::dataformats::PrimaryVertex>& primVertices,
                  const std::vector<o2::InteractionTimeRecord>& mcRecords,
                  std::map<uint64_t, int>& bcsMap);

  uint64_t getTFNumber(const o2::InteractionRecord& tfStartIR, int runNumber);

  template <typename TracksCursorType, typename TracksCovCursorType>
  void addToTracksTable(TracksCursorType& tracksCursor, TracksCovCursorType& tracksCovCursor,
                        const o2::track::TrackParCov& track, int collisionID, int src);

  template <typename TracksExtraCursorType>
  void addToTracksExtraTable(TracksExtraCursorType& tracksExtraCursor);

  template <typename mftTracksCursorType>
  void addToMFTTracksTable(mftTracksCursorType& mftTracksCursor, const o2::mft::TrackMFT& track, int collisionID);

  template <typename MCParticlesCursorType>
  void fillMCParticlesTable(o2::steer::MCKinematicsReader& mcReader, const MCParticlesCursorType& mcParticlesCursor,
                            gsl::span<const o2::MCCompLabel>& mcTruthITS, std::vector<bool>& isStoredITS,
                            gsl::span<const o2::MCCompLabel>& mcTruthMFT, std::vector<bool>& isStoredMFT,
                            gsl::span<const o2::MCCompLabel>& mcTruthTPC, std::vector<bool>& isStoredTPC,
                            TripletsMap_t& toStore);
};

/// create a processor spec
framework::DataProcessorSpec getAODProducerWorkflowSpec(GID::mask_t src, bool useMC, bool fillSVertices);

} // namespace o2::aodproducer

#endif /* O2_AODPRODUCER_WORKFLOW_SPEC */
